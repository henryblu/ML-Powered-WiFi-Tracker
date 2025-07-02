#!/usr/bin/env python3
"""Serial port reader for CSI data."""
from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Tuple

import serial


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
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Open the serial connection and read until cancelled."""
        self._loop = asyncio.get_running_loop()
        self._ser = serial.serial_for_url(self.port, self.baud, timeout=1)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()

    def _read_loop(self) -> None:
        assert self._ser is not None
        assert self._loop is not None
        try:
            while not self._stop.is_set():
                try:
                    line = self._ser.readline()
                except Exception:
                    continue
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                fut = asyncio.run_coroutine_threadsafe(
                    self.queue.put((self.receiver_id, text)), self._loop
                )
                try:
                    fut.result()
                except Exception:
                    pass
        finally:
            self._stop.set()

    async def close(self) -> None:
        """Close the serial connection."""
        self._stop.set()
        if self._ser and self._ser.is_open:
            with contextlib.suppress(Exception):
                self._ser.cancel_read()
            self._ser.close()
        if self._thread and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join)
