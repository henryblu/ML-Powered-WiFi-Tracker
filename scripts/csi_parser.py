#!/usr/bin/env python3
"""Parse CSI lines from the ESP32 receivers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CSIPacket:
    """Represent a CSI packet."""

    mac: str
    seq_ctrl: int
    timestamp: int
    rssi: int
    channel: int
    csi_complex: List[complex]
    csi_raw: str
    receiver_id: str


class Parser:
    """Convert raw text lines into :class:`CSIPacket` objects."""

    def __init__(
        self,
        queue_in: asyncio.Queue[Tuple[str, str]],
        queue_out: asyncio.Queue[CSIPacket],
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out

    @staticmethod
    def _parse_csi(csi_str: str) -> List[complex]:
        numbers = [int(x) for x in csi_str.strip().split() if x]
        complex_vals = []
        for i in range(0, len(numbers), 2):
            if i + 1 < len(numbers):
                complex_vals.append(complex(numbers[i], numbers[i + 1]))
        return complex_vals

    @staticmethod
    def _parse_line(line: str, receiver_id: str) -> Optional[CSIPacket]:
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            if "," in line:
                _, rest = line.split(",", 1)
                line = rest.strip().strip('"')
        if not line.startswith("CSI_DATA"):
            return None
        try:
            prefix, csi_part = line.split("[", 1)
        except ValueError:
            return None
        csi_part = csi_part.rstrip("]")
        csi_raw = csi_part.strip()
        fields = prefix.split(",")
        if len(fields) < 26:
            return None
        mac = fields[2]
        rssi = int(fields[3])
        channel = int(fields[16])
        timestamp = int(fields[18])
        seq_ctrl = int(fields[25])
        csi_complex = Parser._parse_csi(csi_part)
        return CSIPacket(
            mac,
            seq_ctrl,
            timestamp,
            rssi,
            channel,
            csi_complex,
            csi_raw,
            receiver_id,
        )

    async def run(self) -> None:
        try:
            while True:
                receiver_id, line = await self.queue_in.get()
                pkt = self._parse_line(line, receiver_id)
                if pkt:
                    await self.queue_out.put(pkt)
        except asyncio.CancelledError:
            pass
