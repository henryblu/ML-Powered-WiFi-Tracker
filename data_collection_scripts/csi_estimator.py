#!/usr/bin/env python3
"""Angle-of-Arrival estimation and math helpers."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional, Tuple

import numpy as np

from data_collection_scripts.csi_parser import CSIPacket


def wavelength(channel: int) -> float:
    """Return carrier wavelength for a 2.4 GHz Wi-Fi channel in meters."""
    if not 1 <= channel <= 14:
        raise ValueError("channel must be in 1..14")
    freq_ghz = 2.412 + 0.005 * (channel - 1)
    return 0.299792458 / freq_ghz


def weighted_phase_mean(phases: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted mean of unwrapped phases."""
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    return float(np.sum(phases * weights) / w_sum)


def asin_clamped(x: float) -> float:
    """Return ``asin(x)`` with ``x`` clamped to [-1, 1]."""
    return math.asin(max(-1.0, min(1.0, x)))


class AoAEstimator:
    """Estimate Angle-of-Arrival from matched CSI packet pairs."""

    def __init__(
        self,
        queue_in: asyncio.Queue[Tuple[CSIPacket, CSIPacket]],
        queue_out: asyncio.Queue[Tuple[int, str, int, float, int, int, str, str]],
        antenna_dist: float = 0.06,
        cal_vector: Optional[np.ndarray] = None,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.wavelength: Optional[float] = None
        self.antenna_dist = antenna_dist
        self.cal_vector = cal_vector

    def _compute_aoa(self, master_pkt: CSIPacket, worker_pkt: CSIPacket) -> float:
        m = np.asarray(master_pkt.csi_complex, dtype=np.complex64)
        w = np.asarray(worker_pkt.csi_complex, dtype=np.complex64)
        if len(m) == 0 or len(m) != len(w):
            raise ValueError("CSI length mismatch")

        if self.wavelength is None:
            self.wavelength = wavelength(master_pkt.channel)

        delta = np.angle(w * np.conj(m)).astype(np.float32)
        unwrapped = np.unwrap(delta)
        if self.cal_vector is not None and len(self.cal_vector) == len(unwrapped):
            unwrapped = unwrapped - self.cal_vector
        weights = np.abs(m) * np.abs(w)
        phi = weighted_phase_mean(unwrapped, weights)
        arg = (self.wavelength / (2 * np.pi * self.antenna_dist)) * phi
        return np.degrees(asin_clamped(float(arg)))

    async def run(self) -> None:
        try:
            while True:
                master_pkt, worker_pkt = await self.queue_in.get()
                try:
                    aoa = self._compute_aoa(master_pkt, worker_pkt)
                    result = (
                        master_pkt.timestamp,
                        master_pkt.mac,
                        master_pkt.seq_ctrl,
                        aoa,
                        master_pkt.rssi,
                        worker_pkt.rssi,
                        master_pkt.csi_raw,
                        worker_pkt.csi_raw,
                    )
                    await self.queue_out.put(result)
                except Exception as exc:  # pragma: no cover - logging only
                    logging.warning(f"AoA computation error: {exc}")
        except asyncio.CancelledError:
            pass
