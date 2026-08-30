import importlib

import poker_vision

EXPECTED_SUBMODULES = [
    "capture",
    "calibration",
    "detection",
    "tracking",
    "assignment",
    "state",
    "export",
    "debug",
    "tools",
    "runner",
]


def test_package_has_version():
    assert poker_vision.__version__ == "0.1.0"


def test_expected_submodules_are_importable():
    for name in EXPECTED_SUBMODULES:
        importlib.import_module(f"poker_vision.{name}")
