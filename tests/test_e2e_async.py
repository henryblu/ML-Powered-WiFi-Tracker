import asyncio

import pytest

from scripts.csi_estimator import AoAEstimator
from scripts.csi_matcher import PairMatcher
from scripts.csi_parser import Parser


@pytest.mark.asyncio
async def test_e2e_pipeline():
    fields = ["CSI_DATA", "0", "aa", "-30"] + ["0"] * 21 + ["1", "0"]
    fields[16] = "1"
    master_line = ",".join(fields) + "[1 0]"
    worker_line = ",".join(fields) + "[0 1]"

    raw_q = asyncio.Queue()
    packet_q = asyncio.Queue()
    pair_q = asyncio.Queue()
    result_q = asyncio.Queue()

    parser = Parser(raw_q, packet_q)
    matcher = PairMatcher(packet_q, pair_q)
    estimator = AoAEstimator(pair_q, result_q)

    async def feeder():
        await raw_q.put(("master", master_line))
        await raw_q.put(("worker", worker_line))

    tasks = [
        asyncio.create_task(parser.run()),
        asyncio.create_task(matcher.run()),
        asyncio.create_task(estimator.run()),
        asyncio.create_task(feeder()),
    ]
    await asyncio.wait(tasks, timeout=0.5)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    ts, mac, seq, aoa, iq_m, iq_w = await asyncio.wait_for(result_q.get(), timeout=1)
    assert mac == "aa"
    assert seq == 1
    assert aoa == pytest.approx(31, abs=1)
    assert iq_m == "1 0"
    assert iq_w == "0 1"
