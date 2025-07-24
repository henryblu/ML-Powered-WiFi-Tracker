import numpy as np
import pytest

from data_collection_scripts.csi_estimator import (
    asin_clamped,
    wavelength,
    weighted_phase_mean,
    AoAEstimator,
)
from data_collection_scripts.csi_parser import CSIPacket


def test_math_helpers():
    assert abs(wavelength(1) - 0.1249166666) < 1e-3
    phases = np.array([np.pi / 2, np.pi / 2], dtype=np.float32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    assert weighted_phase_mean(phases, weights) == pytest.approx(np.pi / 2)
    assert asin_clamped(0.5) == pytest.approx(np.arcsin(0.5))
    assert asin_clamped(2) == pytest.approx(np.arcsin(1))

    # synthetic AoA estimation
    pkt_m = CSIPacket(
        mac="aa",
        seq_ctrl=1,
        timestamp=0,
        rssi=-30,
        channel=1,
        csi_complex=[1 + 0j],
        csi_raw="1 0",
        receiver_id="master",
    )
    pkt_w = CSIPacket(
        mac="aa",
        seq_ctrl=1,
        timestamp=0,
        rssi=-30,
        channel=1,
        csi_complex=[0 + 1j],
        csi_raw="0 1",
        receiver_id="worker",
    )
    estimator = AoAEstimator(None, None)
    aoa = estimator._compute_aoa(pkt_m, pkt_w)
    assert aoa == pytest.approx(31, abs=1)
