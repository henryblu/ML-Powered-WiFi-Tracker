#!/usr/bin/env python3
"""Angle-of-Arrival estimation with corrected logic using raw CSI string parsing."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Tuple

import numpy as np

from data_collection_scripts.csi_parser import CSIPacket  # Presumed available


def wavelength(channel: int) -> float:
    """Return carrier wavelength for a 2.4 GHz Wi-Fi channel in meters."""
    if not 1 <= channel <= 14:
        raise ValueError("channel must be in 1..14")
    freq_ghz = 2.412 + 0.005 * (channel - 1)
    return 0.299792458 / freq_ghz


def get_phase_array(csi_str: str) -> np.ndarray:
    """Extract phase values from raw CSI string data."""
    csi_str = csi_str.strip("[]")
    raw = np.array([int(x) for x in csi_str.split()])
    I_values = raw[::2]
    Q_values = raw[1::2]
    return np.angle(I_values + 1j * Q_values)


def asin_clamped(x: float) -> float:
    """Return asin(x), clamped to [-1, 1]."""
    return math.asin(max(-1.0, min(1.0, x)))


class AoAEstimator:
    """Estimate Angle-of-Arrival from matched CSI packet pairs."""

    def __init__(
        self,
        queue_in: asyncio.Queue[Tuple[CSIPacket, CSIPacket]],
        queue_out: asyncio.Queue[Tuple[int, str, int, float, int, int, str, str]],
        antenna_dist: float = 0.06,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.antenna_dist = antenna_dist

    def _compute_aoa(self, master_pkt: CSIPacket, worker_pkt: CSIPacket) -> float:
        """Compute AoA using extracted phase difference from CSI raw strings."""
        try:
            phase1 = get_phase_array(master_pkt.csi_raw)
            phase2 = get_phase_array(worker_pkt.csi_raw)

            min_len = min(len(phase1), len(phase2))
            phase1 = phase1[:min_len]
            phase2 = phase2[:min_len]

            phase_diff = np.unwrap(phase2 - phase1)
            mean_phase_diff = np.mean(phase_diff)

            freq_hz = 2.412e9 + (master_pkt.channel - 1) * 5e6
            lam = 3e8 / freq_hz  # wavelength in meters

            sin_theta = (lam * mean_phase_diff) / (2 * np.pi * self.antenna_dist)
            sin_theta = np.clip(sin_theta, -1, 1)

            aoa_rad = np.arcsin(sin_theta)
            return float(np.degrees(aoa_rad))
        except Exception as e:
            raise ValueError(f"Failed AoA computation: {e}")

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
                except Exception as exc:
                    logging.warning(f"AoA computation error: {exc}")
        except asyncio.CancelledError:
            pass
