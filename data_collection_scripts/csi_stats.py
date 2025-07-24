#!/usr/bin/env python3
"""Statistics reporting coroutine."""
from __future__ import annotations

import asyncio

from data_collection_scripts.csi_matcher import PairMatcher


async def stats_loop(matcher: PairMatcher) -> None:
    prev = 0
    try:
        while True:
            await asyncio.sleep(5)
            total = matcher.master_count + matcher.worker_count
            lines = total - prev
            prev = total
            tp = lines / 5
            print(
                f"Master: {matcher.master_count} Worker: {matcher.worker_count} "
                f"Paired: {matcher.paired} Dropped: {matcher.dropped} "
                f"Throughput: {tp:.1f} lines/s"
            )
    except asyncio.CancelledError:
        pass
