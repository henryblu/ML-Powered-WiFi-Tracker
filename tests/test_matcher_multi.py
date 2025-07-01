import asyncio

import pytest

from scripts.csi_matcher import PairMatcher
from scripts.csi_parser import CSIPacket


@pytest.mark.asyncio
async def test_multi_pair_sorted():
    in_q = asyncio.Queue()
    out_q = asyncio.Queue()
    matcher = PairMatcher(in_q, out_q, ts_window=10)
    task = asyncio.create_task(matcher.run())

    masters = [CSIPacket("aa", 1, t, -30, 1, [], "", "master") for t in (0, 10, 20)]
    workers = [CSIPacket("aa", 1, t, -30, 1, [], "", "worker") for t in (1, 11, 21)]

    # feed masters then workers in reverse order
    for pkt in masters:
        await in_q.put(pkt)
    for pkt in reversed(workers):
        await in_q.put(pkt)

    for _ in range(10):
        if out_q.qsize() >= 3:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    results = [out_q.get_nowait() for _ in range(3)]
    ts_order = [m.timestamp for m, _ in results]
    assert ts_order == [0, 10, 20]
    assert matcher.dropped == 0
