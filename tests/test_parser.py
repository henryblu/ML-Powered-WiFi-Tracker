import asyncio

import pytest

from scripts.csi_parser import Parser, CSIPacket


@pytest.mark.asyncio
async def test_parse_line():
    fields = (
        [
            "CSI_DATA",
            "0",
            "aa:bb:cc:dd:ee:ff",
            "-30",
        ]
        + ["0"] * 21
        + ["42", "0"]
    )
    fields[16] = "1"
    sample = ",".join(fields) + "[1 2 3 4]"
    q_in: asyncio.Queue = asyncio.Queue()
    q_out: asyncio.Queue = asyncio.Queue()
    parser = Parser(q_in, q_out)
    await q_in.put(("master", sample))
    await asyncio.sleep(0)
    task = asyncio.create_task(parser.run())
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    pkt = q_out.get_nowait()
    assert isinstance(pkt, CSIPacket)
    assert pkt.mac == "aa:bb:cc:dd:ee:ff"
    assert pkt.seq_ctrl == 42
    assert pkt.channel == 1
    assert pkt.csi_complex == [complex(1, 2), complex(3, 4)]
