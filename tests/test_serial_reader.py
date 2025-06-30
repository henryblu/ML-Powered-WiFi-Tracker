import asyncio

import pytest

from scripts.csi_reader import SerialReader


@pytest.mark.asyncio
async def test_serial_reader_loopback():
    q = asyncio.Queue()
    reader = SerialReader('loop://', 9600, 'm', q)
    task = asyncio.create_task(reader.start())
    await asyncio.sleep(0.1)
    assert reader._ser is not None
    reader._ser.write(b'test\n')
    receiver, line = await asyncio.wait_for(q.get(), timeout=1)
    assert receiver == 'm'
    assert line == 'test'
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await reader.close()
