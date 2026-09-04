import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "poker_vision"
CONFIG_FILE = SRC_ROOT / "config.py"

# The only nodes in config.py allowed to mention "cuda": the DeviceType enum
# (which must define the reserved member) and the validator that rejects it.
ALLOWED_NODE_NAMES = {"DeviceType", "_reject_cuda_device"}


def _cuda_lines(path: Path) -> list[int]:
    return [
        lineno
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "cuda" in line.lower()
    ]


def _allowed_config_lines() -> set[int]:
    tree = ast.parse(CONFIG_FILE.read_text(), filename=str(CONFIG_FILE))
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in ALLOWED_NODE_NAMES:
            allowed.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return allowed


def test_no_cuda_references_outside_config():
    offenders = {
        path: lines
        for path in SRC_ROOT.rglob("*.py")
        if path != CONFIG_FILE and (lines := _cuda_lines(path))
    }
    assert not offenders, f"unexpected 'cuda' references: {offenders}"


def test_config_cuda_references_confined_to_device_guard():
    allowed = _allowed_config_lines()
    source_lines = CONFIG_FILE.read_text().splitlines()
    stray = [
        lineno
        for lineno in _cuda_lines(CONFIG_FILE)
        if lineno not in allowed and not source_lines[lineno - 1].strip().startswith("#")
    ]
    assert not stray, (
        f"'cuda' referenced in config.py outside DeviceType/_reject_cuda_device: {stray}"
    )


def test_config_still_guards_cuda_device():
    assert _cuda_lines(CONFIG_FILE), "expected config.py to reject the reserved cuda device"
