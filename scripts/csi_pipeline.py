#!/usr/bin/env python3
"""Main entry point for the CSI AoA pipeline."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

# Make sibling modules importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts.csi_reader import SerialReader
from scripts.csi_parser import Parser
from scripts.csi_matcher import PairMatcher
from scripts.csi_estimator import AoAEstimator
from scripts.csi_logger import CSILogger
from scripts.csi_stats import stats_loop


@dataclass
class Settings:
    port_master: str = ""
    port_worker: str = ""
    baud: int = 921600
    output: str = "aoa_log.csv"
    queue_size: int = 5000
    antenna_distance_m: float = 0.06
    gc_interval: float = 0.01
    match_timeout: float = 1.0
    flush_bytes: int = 8192
    calibration: Optional[str] = None
    stats: bool = False


def env_or_cli(args: argparse.Namespace) -> Settings:
    data = {f.name: getattr(Settings(), f.name) for f in fields(Settings)}
    prefix = "CSI_PIPELINE_"
    for name in data:
        env = os.getenv(prefix + name.upper())
        if env is not None:
            value = data[name]
            if isinstance(value, bool):
                data[name] = env.lower() in {"1", "true", "yes"}
            elif isinstance(value, int):
                data[name] = int(env)
            elif isinstance(value, float):
                data[name] = float(env)
            else:
                data[name] = env
    cli_map = {
        "port_master": args.port_master,
        "port_worker": args.port_worker,
        "baud": args.baud,
        "output": args.output,
        "calibration": args.calibration,
        "stats": args.stats,
    }
    for k, v in cli_map.items():
        if v is not None:
            data[k] = v
    settings = Settings(**data)
    if not settings.port_master or not settings.port_worker:
        raise SystemExit("port_master and port_worker must be specified")
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSI AoA pipeline")
    parser.add_argument("--port-master")
    parser.add_argument("--port-worker")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--output")
    parser.add_argument("--calibration")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def load_calibration(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        cal = np.load(path)
        if cal.ndim != 1:
            raise ValueError("calibration vector must be 1-D")
        return cal.astype(np.float32)
    except Exception as exc:
        logging.warning(f"Failed to load calibration file: {exc}")
        return None


async def main_async(opts: Settings) -> None:
    raw_queue: asyncio.Queue = asyncio.Queue(maxsize=opts.queue_size)
    packet_queue: asyncio.Queue = asyncio.Queue(maxsize=opts.queue_size)
    pair_queue: asyncio.Queue = asyncio.Queue(maxsize=opts.queue_size)
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=opts.queue_size)

    reader_master = SerialReader(opts.port_master, opts.baud, "master", raw_queue)
    reader_worker = SerialReader(opts.port_worker, opts.baud, "worker", raw_queue)
    parser = Parser(raw_queue, packet_queue)
    matcher = PairMatcher(
        packet_queue,
        pair_queue,
        gc_interval=opts.gc_interval,
        timeout=opts.match_timeout,
    )
    cal_vector = load_calibration(opts.calibration)
    estimator = AoAEstimator(
        pair_queue,
        result_queue,
        antenna_dist=opts.antenna_distance_m,
        cal_vector=cal_vector,
    )
    logger = CSILogger(opts.output, result_queue, flush_bytes=opts.flush_bytes)

    tasks = [
        asyncio.create_task(reader_master.start()),
        asyncio.create_task(reader_worker.start()),
        asyncio.create_task(parser.run()),
        asyncio.create_task(matcher.run()),
        asyncio.create_task(estimator.run()),
        asyncio.create_task(logger.run()),
    ]
    if opts.stats:
        tasks.append(asyncio.create_task(stats_loop(matcher)))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    logger.close()
    with contextlib.suppress(Exception):
        await reader_master.close()
    with contextlib.suppress(Exception):
        await reader_worker.close()


def main() -> None:
    args = parse_args()
    opts = env_or_cli(args)
    logging.basicConfig(filename="warnings.log", level=logging.WARNING)
    try:
        asyncio.run(main_async(opts))
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
