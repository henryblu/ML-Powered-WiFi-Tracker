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
        ts_window: int | None = None,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.gc_interval = gc_interval
        self.timeout = timeout
        self.ts_window = ts_window
        self.pending: Dict[
            Tuple[str, int], Dict[str, list[tuple[CSIPacket, float]]]
        ] = {}
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
                    for role in list(v):
                        pkt_list = v[role]
                        while pkt_list and now - pkt_list[0][1] > self.timeout:
                            logging.warning(f"Pair timeout for {k}")
                            pkt_list.pop(0)
                            self.dropped += 1
                        if not pkt_list:
                            v.pop(role)
                    if not v:
                        self.pending.pop(k, None)
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
                pkt_list = entry.setdefault(pkt.receiver_id, [])
                pkt_list.append((pkt, now))
                if pkt.receiver_id == "master":
                    self.master_count += 1
                else:
                    self.worker_count += 1
                if "master" in entry and "worker" in entry:
                    masters = entry["master"]
                    workers = entry["worker"]
                    while masters and workers:
                        m_pkt, _ = masters[0]
                        w_pkt, _ = workers[0]
                        diff = abs(m_pkt.timestamp - w_pkt.timestamp)
                        if self.ts_window is None or diff <= self.ts_window:
                            masters.pop(0)
                            workers.pop(0)
                            self.paired += 1
                            await self.queue_out.put((m_pkt, w_pkt))
                        else:
                            if m_pkt.timestamp < w_pkt.timestamp:
                                masters.pop(0)
                            else:
                                workers.pop(0)
                            self.dropped += 1
                    if not masters and not workers:
                        self.pending.pop(key, None)
        except asyncio.CancelledError:
            pass
        finally:
            if self._gc_task:
                self._gc_task.cancel()
                await asyncio.gather(self._gc_task, return_exceptions=True)
