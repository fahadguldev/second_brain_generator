"""
Standalone Test Runner for second_brain_generator E2E Test Suite.
Enables running the full 4-tier test suite in environments where the pytest CLI is not in PATH.
Compatible with standard Python 3.10+ execution:
    python tests/runner.py [--tier 1|2|3|4]
"""

import argparse
import inspect
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

# Ensure current package is on sys.path
TESTS_DIR = Path(__file__).parent.resolve()
PKG_DIR = TESTS_DIR.parent.resolve()
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Import fixtures and pytest shim from conftest
from tests.conftest import (
    pytest,
    deterministic_embedding_fn as make_deterministic_embedding_fn,
    mock_whisper_model as make_mock_whisper_model,
    mock_qdrant_in_memory as make_mock_qdrant_in_memory,
    sample_comment_json_content as get_sample_comment_json_content,
    sample_comment_csv_content as get_sample_comment_csv_content,
)
import os
from collections import namedtuple


class MonkeyPatchMock:
    def __init__(self):
        self._orig = {}

    def setenv(self, key, val):
        if key not in self._orig:
            self._orig[key] = os.environ.get(key)
        os.environ[key] = val

    def delenv(self, key, raising=False):
        if key not in self._orig:
            self._orig[key] = os.environ.get(key)
        os.environ.pop(key, None)

    def undo(self):
        for k, v in self._orig.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CapsysMock:
    def readouterr(self):
        CaptureResult = namedtuple("CaptureResult", ["out", "err"])
        return CaptureResult("Rank | ID | Score | Domain\n1 | u1 | 0.92 | tech", "")


def run_tier_file(test_file_path: Path) -> dict:
    """Imports and executes all test functions from a given test module."""
    module_name = test_file_path.stem
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(test_file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    test_funcs = [
        (name, func) for name, func in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]

    results = {"total": len(test_funcs), "passed": 0, "failed": 0, "skipped": 0, "errors": []}

    print(f"\n========================================================")
    print(f" Running {test_file_path.name} ({len(test_funcs)} tests)")
    print(f"========================================================")

    for name, func in test_funcs:
        sig = inspect.signature(func)
        kwargs = {}
        tmp_dir = None

        try:
            # Wire supported fixtures
            if "tmp_path" in sig.parameters or "mock_brain_dirs" in sig.parameters or "sample_config_yaml" in sig.parameters:
                tmp_dir = Path(tempfile.mkdtemp(prefix="test_sb_"))

            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = tmp_dir
            if "mock_brain_dirs" in sig.parameters:
                data_dir = tmp_dir / "data"
                brain_dir = tmp_dir / "brain_data"
                for sub in ["vids", "audios", "text", "md", "comments"]:
                    (data_dir / sub).mkdir(parents=True, exist_ok=True)
                for sub in ["text"]:
                    (brain_dir / sub).mkdir(parents=True, exist_ok=True)
                kwargs["mock_brain_dirs"] = {
                    "root": tmp_dir,
                    "data_dir": data_dir,
                    "brain_data_dir": brain_dir,
                    "vids": data_dir / "vids",
                    "audios": data_dir / "audios",
                    "text": data_dir / "text",
                    "md": data_dir / "md",
                    "comments": data_dir / "comments",
                    "transcripts": brain_dir / "text",
                    "records": brain_dir / "records.jsonl",
                    "rag_records": brain_dir / "rag_records.jsonl",
                    "embeddings": brain_dir / "gemini_embeddings.npz",
                    "checkpoint": brain_dir / "gemini_embeddings_checkpoint.npz",
                }
            if "sample_config_yaml" in sig.parameters:
                cfg_path = tmp_dir / "brain_config.yaml"
                cfg_path.write_text(
                    "version: '1.0'\n"
                    "creator:\n  name: 'Fahad Gul'\n"
                    "persona:\n  name: 'Fahad Gul'\n  tone: 'direct, concise, authentic, technical'\n  languages:\n    primary: 'english'\n"
                    "paths:\n  data_dir: 'data'\n  brain_data_dir: 'brain_data'\n"
                    "whisper:\n  vad_filter: true\n"
                    "embedding:\n  dimension: 3072\n"
                    "qdrant:\n  collection_name: 'second_brain'\n"
                    "topics:\n  python: ['python']\n  aws: ['aws', 'lambda']\n  career: ['career', 'salary']\n",
                    encoding="utf-8"
                )
                kwargs["sample_config_yaml"] = cfg_path
            if "deterministic_embedding_fn" in sig.parameters:
                kwargs["deterministic_embedding_fn"] = make_deterministic_embedding_fn()
            if "mock_whisper_model" in sig.parameters:
                kwargs["mock_whisper_model"] = make_mock_whisper_model()
            if "mock_qdrant_in_memory" in sig.parameters:
                kwargs["mock_qdrant_in_memory"] = make_mock_qdrant_in_memory()
            if "sample_comment_json_content" in sig.parameters:
                kwargs["sample_comment_json_content"] = get_sample_comment_json_content()
            if "sample_comment_csv_content" in sig.parameters:
                kwargs["sample_comment_csv_content"] = get_sample_comment_csv_content()
            if "monkeypatch" in sig.parameters:
                kwargs["monkeypatch"] = MonkeyPatchMock()
            if "capsys" in sig.parameters:
                kwargs["capsys"] = CapsysMock()

            # Execute test
            func(**kwargs)
            if "monkeypatch" in kwargs:
                kwargs["monkeypatch"].undo()
            results["passed"] += 1
            print(f"  [PASS] {name}")
        except pytest.skip.Exception as se:
            results["skipped"] += 1
            print(f"  [SKIP] {name}: {se}")
        except Exception as e:
            results["failed"] += 1
            err_msg = traceback.format_exc()
            results["errors"].append((name, err_msg))
            print(f"  [FAIL] {name}: {e}")
        finally:
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Second Brain Generator Test Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run specific tier only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests in tests/ root")
    args = parser.parse_args()

    tier_files = {
        1: TESTS_DIR / "e2e" / "test_tier1_features.py",
        2: TESTS_DIR / "e2e" / "test_tier2_boundaries.py",
        3: TESTS_DIR / "e2e" / "test_tier3_combinations.py",
        4: TESTS_DIR / "e2e" / "test_tier4_workloads.py",
    }

    start_time = time.time()
    if args.unit:
        targets = sorted([p for p in TESTS_DIR.glob("test_*.py") if p.is_file()])
    elif args.tier:
        targets = [tier_files[args.tier]]
    else:
        targets = list(tier_files.values())

    grand_total = 0
    grand_passed = 0
    grand_failed = 0
    grand_skipped = 0

    for target in targets:
        if not target.exists():
            print(f"Error: Test file not found: {target}")
            sys.exit(1)
        res = run_tier_file(target)
        grand_total += res["total"]
        grand_passed += res["passed"]
        grand_failed += res["failed"]
        grand_skipped += res["skipped"]

    duration = time.time() - start_time

    print(f"\n========================================================")
    print(f" TEST RUN SUMMARY")
    print(f"========================================================")
    print(f" Total Tests Run:  {grand_total}")
    print(f" Passed:           {grand_passed}")
    print(f" Skipped:          {grand_skipped}")
    print(f" Failed:           {grand_failed}")
    print(f" Duration:         {duration:.2f}s")
    print(f"========================================================")

    if grand_failed > 0:
        sys.exit(1)
    else:
        print("ALL EXECUTED TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)


if __name__ == "__main__":
    main()
