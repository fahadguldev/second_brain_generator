#!/usr/bin/env python3
"""
Root entry-point proxy for the Ganjiswag AI Second Brain testing pipeline.
Enables running `python testing.py` directly from the workspace root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
BRAIN_GEN_DIR = SCRIPT_DIR / "brain_generator"

# Ensure brain_generator package and root are on sys.path
for p in [str(BRAIN_GEN_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

target_script = BRAIN_GEN_DIR / "testing.py"
if not target_script.exists():
    raise FileNotFoundError(f"Underlying testing pipeline script not found at: {target_script}")

spec = importlib.util.spec_from_file_location("ganjiswag_testing", target_script)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load spec for: {target_script}")

module = importlib.util.module_from_spec(spec)
sys.modules["ganjiswag_testing"] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.main()
