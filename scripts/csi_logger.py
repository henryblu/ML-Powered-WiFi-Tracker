#!/usr/bin/env python3
"""CSV logger with file rotation."""
from __future__ import annotations

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from typing import Tuple


class CSILogger:
    """Write AoA results to a rotating CSV file."""

    def __init__(
        self,
        base_path: str,
        queue_in: asyncio.Queue[Tuple[int, str, int, float]],
        flush_bytes: int = 8192,
    ):
        self.base_path = base_path
        self.queue_in = queue_in
        self.flush_bytes = flush_bytes
        self.handler = RotatingFileHandler(
            filename=self.base_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger = logging.getLogger("csi-logger")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        if self.handler.stream.tell() == 0:
            self.logger.info("timestamp,mac,seq_ctrl,aoa")
        self._last_flush = self.handler.stream.tell()

    async def run(self) -> None:
        try:
            while True:
                timestamp, mac, seq_ctrl, aoa = await self.queue_in.get()
                self.logger.info(f"{timestamp},{mac},{seq_ctrl},{aoa:.2f}")
                if self.handler.stream.tell() - self._last_flush > self.flush_bytes:
                    self.handler.flush()
                    self._last_flush = self.handler.stream.tell()
        except asyncio.CancelledError:
            pass

    def close(self) -> None:
        for h in list(self.logger.handlers):
            h.flush()
            h.close()
            self.logger.removeHandler(h)
