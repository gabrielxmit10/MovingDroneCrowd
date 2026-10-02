"""Exercise notebook routing and the old-Colab compatibility fix without a GPU."""

import contextlib
import datetime
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "colab_steerer_mdc_test.ipynb").read_text(encoding="utf-8"))
CELLS = {cell.get("id"): "".join(cell["source"]) for cell in NOTEBOOK["cells"]}


class NotebookValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        repo = root / "repo"
        repo.mkdir()
        # This reproduces the obsolete source still present in the user's Colab.
        self.evaluator = repo / "evaluate_steerer_mdc.py"
        self.evaluator.write_text(
            "def evaluate(torch, device):\n"
            "    torch.cuda.reset_peak_memory_stats(device)\n",
            encoding="utf-8",
        )
        self.calls = []
        self.env = dict(
            Path=Path, json=json, shutil=shutil, sys=sys, datetime=datetime,
            pd=SimpleNamespace(DataFrame=lambda value: value), display=lambda value: None,
            ACTION="evaluate", TEST_SPLIT="val.txt", CONFIRM_FULL_TEST=False,
            RUN_DIR=root / "run", RUN_NAME="validation_reference", REPO_DIR=repo,
            DRIVE_PROJECT=root, DATASET_ROOT=root / "data", LOCAL_CHECKPOINT=root / "model.pth",
            DEVICE="cuda:0", SMOKE_SAMPLES=3, EVAL_MAX_SAMPLES=0,
            FRAME_STRIDE=1, MAX_LONG_SIDE=1920, MAX_SHORT_SIDE=1080,
            SEED=3035, SYNC_EVERY=10, DENSITY_MAP_LIMIT=20,
            SAVE_DENSITY_MAPS=False, USE_AMP=False, RESUME_EXISTING=True,
            run_command=self.run_command,
        )

    def run_command(self, command):
        self.calls.append([str(value) for value in command])
        namespace = {}
        exec(self.evaluator.read_text(encoding="utf-8"), namespace)

        def invalid_reset(device):
            raise RuntimeError("Invalid device argument")

        namespace["evaluate"](SimpleNamespace(cuda=SimpleNamespace(
            reset_peak_memory_stats=invalid_reset,
        )), "cuda:0")
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    def run_action(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS["304a81ac"], self.env)

    def test_validation_repairs_old_evaluator_and_is_safe_to_rerun(self):
        self.run_action()
        repaired = self.evaluator.read_text(encoding="utf-8")
        self.run_action()
        self.assertEqual(self.evaluator.read_text(encoding="utf-8"), repaired)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.env["CURRENT_RESULT_DIR"], self.env["RUN_DIR"] / "val")
        self.assertEqual((self.env["CURRENT_RESULT_DIR"] / "evaluator_used.py").read_text(encoding="utf-8"), repaired)
        args = self.calls[-1]
        self.assertEqual(args[args.index("--split-file") + 1], "val.txt")
        self.assertEqual(args[args.index("--max-samples") + 1], "0")
        self.assertIn("--resume", args)
        latest = json.loads((self.env["DRIVE_PROJECT"] / "latest_action.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["split_file"], "val.txt")

    def test_test_requires_confirmation_and_retains_test_directory(self):
        self.env["TEST_SPLIT"] = "test.txt"
        with self.assertRaisesRegex(RuntimeError, "CONFIRM_FULL_TEST"):
            self.run_action()
        self.assertEqual(self.calls, [])
        self.env["CONFIRM_FULL_TEST"] = True
        self.run_action()
        self.assertEqual(self.env["CURRENT_RESULT_DIR"].name, "test")

    def test_smoke_keeps_sample_limit_and_directory(self):
        self.env["ACTION"] = "smoke"
        self.run_action()
        self.assertEqual(self.env["CURRENT_RESULT_DIR"].name, "smoke")
        args = self.calls[-1]
        self.assertEqual(args[args.index("--max-samples") + 1], "3")

    def test_inspect_does_not_invoke_model(self):
        self.env.update(ACTION="inspect", PREFLIGHT_DIR=self.env["RUN_DIR"] / "preflight")
        self.run_action()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.env["CURRENT_RESULT_DIR"].name, "preflight")

    def test_every_code_cell_compiles(self):
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell.get("id", "cell"), "exec")


if __name__ == "__main__":
    unittest.main()
