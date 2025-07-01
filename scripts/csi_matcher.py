#!/usr/bin/env python3
"""Match CSI packets from master and worker devices."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Tuple

from scripts.csi_parser import CSIPacket


class PairMatcher:
    """Pair CSI packets with the same MAC and sequence number."""

    def __init__(
        self,
        queue_in: asyncio.Queue[CSIPacket],
        queue_out: asyncio.Queue[Tuple[CSIPacket, CSIPacket]],
        gc_interval: float = 0.01,
        timeout: float = 1.0,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.gc_interval = gc_interval
        self.timeout = timeout
        self.pending: Dict[Tuple[str, int], Dict[str, tuple[CSIPacket, float]]] = {}
        self.master_count = 0
        self.worker_count = 0
        self.paired = 0
        self.dropped = 0
        self._gc_task: asyncio.Task | None = None

    async def _gc(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.gc_interval)
                now = time.monotonic()
                for k in tuple(self.pending):
                    v = self.pending[k]
                    ts_list = [t for (_, t) in v.values()]
                    if ts_list and now - min(ts_list) > self.timeout:
                        logging.warning(f"Pair timeout for {k}")
                        self.pending.pop(k, None)
                        self.dropped += 1
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        self._gc_task = asyncio.create_task(self._gc())
        try:
            while True:
                pkt = await self.queue_in.get()
                key = (pkt.mac, pkt.seq_ctrl)
                now = time.monotonic()
                entry = self.pending.setdefault(key, {})
                entry[pkt.receiver_id] = (pkt, now)
                if pkt.receiver_id == "master":
                    self.master_count += 1
                else:
                    self.worker_count += 1
                if "master" in entry and "worker" in entry:
                    pair = (entry["master"][0], entry["worker"][0])
                    self.pending.pop(key, None)
                    self.paired += 1
                    await self.queue_out.put(pair)
        except asyncio.CancelledError:
            pass
        finally:
            if self._gc_task:
                self._gc_task.cancel()
                await asyncio.gather(self._gc_task, return_exceptions=True)
