#!/usr/bin/env python3
"""Serial port reader for CSI data."""
from __future__ import annotations

import asyncio
from typing import Tuple

import serial_asyncio


class SerialReader:
    """Read lines from a serial port and put them into an async queue."""

    def __init__(
        self,
        port: str,
        baud: int,
        receiver_id: str,
        queue: asyncio.Queue[Tuple[str, str]],
    ):
        self.port = port
        self.baud = baud
        self.receiver_id = receiver_id
        self.queue = queue
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Open the serial connection and read until cancelled."""
        try:
            self._reader, self._writer = await serial_asyncio.open_serial_connection(
                url=self.port, baudrate=self.baud, newline=b"\n"
            )
        except TypeError:
            # PySerial < 3.5 does not accept ``newline``
            self._reader, self._writer = await serial_asyncio.open_serial_connection(
                url=self.port, baudrate=self.baud
            )
        try:
            await self._read_loop()
        except asyncio.CancelledError:
            pass

    async def _read_loop(self) -> None:
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                await self.queue.put((self.receiver_id, text))
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        """Close the serial connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
