"""Standard-library regression for public-candidate preflight and local output metadata.
Run: python -B mpm/src/tests/test_public_candidate.py
No physical simulation or external publication is performed.
"""
import ast
import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
from tools import u4_check as C, u4_joint as J

CONFIG_REL = "scenes/configs/coupled_u4.json"
BASE = json.loads((WORK / "tools/u4_cases.json").read_text(encoding="utf-8"))
CONFIG = json.loads((WORK / CONFIG_REL).read_text(encoding="utf-8"))
TF = CONFIG["contact"]["surface"]["transform"]
PINS = J.read_pins()


def negatives():
    cases = []
    for key, value in [("R", [[1,0,0],[0,1,0],[0,0,1]]), ("c", [0,0,0]),
                       ("C", [0,0,0]), ("status", "other-contract"),
                       ("user_signoff", "not-an-approval")]:
        t = copy.deepcopy(TF); t[key] = value; cases.append((key, t))
    t = copy.deepcopy(TF); t["R"][0][0] = True; cases.append(("boolean-coordinate", t))
    t = copy.deepcopy(TF); del t["c"]; cases.append(("missing-c", t))
    return cases


class PublicCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mpm-public-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.work = self.root / "src"
        for name in ["tools/u4_cases.json", "tools/u4_check.py", "tests/test_adapter.py", CONFIG_REL]:
            p = self.work / name; p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORK / name, p)

    def args(self):
        from argparse import Namespace
        return Namespace(mesh_dir=None, inte=None, xpbd_root=None)

    def preflight(self):
        with patch.object(J, "WORK", self.work):
            return J.find_inputs(self.args(), PINS)

    def write_transform(self, value):
        config = copy.deepcopy(CONFIG); config["contact"]["surface"]["transform"] = value
        (self.work / CONFIG_REL).write_text(json.dumps(config), encoding="utf-8")

    def test_predicate_positive_and_each_rejection(self):
        self.assertTrue(C.matches_public_transform(TF))
        self.assertFalse(C.matches_public_transform(None))
        for name, tf in negatives():
            with self.subTest(case=name): self.assertFalse(C.matches_public_transform(tf))

    def test_input_preflight_positive_and_each_rejection(self):
        rec, found = self.preflight()
        self.assertIsNotNone(found, rec); self.assertTrue(rec["public_transform_preflight"]["ok"])
        for name, tf in negatives():
            with self.subTest(case=name):
                self.write_transform(tf); rec, found = self.preflight()
                self.assertIsNone(found); self.assertFalse(rec["public_transform_preflight"]["ok"])
                self.assertTrue(any("public transform preflight" in p for p in rec["problems"]))

    def test_missing_or_malformed_config_fails_closed(self):
        p = self.work / CONFIG_REL
        for content in ["{bad json", "{}"]:
            p.write_text(content, encoding="utf-8"); rec, found = self.preflight()
            self.assertIsNone(found); self.assertFalse(rec["public_transform_preflight"]["ok"])
        p.unlink(); self.assertIsNone(self.preflight()[1])

    def test_only_locked_package_config_is_accepted(self):
        for path in ["../outside.json", str(self.root / "outside.json"), "scenes/configs/other.json"]:
            plan = copy.deepcopy(BASE); plan["config"] = path
            (self.work / "tools/u4_cases.json").write_text(json.dumps(plan), encoding="utf-8")
            rec, found = self.preflight(); self.assertIsNone(found)
            self.assertIn("fixed package scene", rec["public_transform_preflight"]["error"])

    def test_invalid_input_stops_before_any_process(self):
        self.write_transform(negatives()[0][1])
        with patch.object(J, "WORK", self.work), patch.object(J, "check_lock", return_value={"ok": True}), \
             patch.object(J, "read_pins", return_value=PINS), \
             patch.object(J, "run_step", side_effect=AssertionError("a process started")) as run_step, \
             patch.object(sys, "argv", ["runner", "--results", str(self.root / "results")]):
            self.assertEqual(J.main(), 4); run_step.assert_not_called()
        summary = json.loads(next((self.root / "results").glob("run-*/run_summary.json")).read_text())
        self.assertEqual(summary["steps"], [])

    def test_result_gate_keeps_the_same_predicate(self):
        tree = ast.parse((WORK / "tools/u4_check.py").read_text(encoding="utf-8"))
        gate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "g_motion")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "matches_public_transform" for n in ast.walk(gate)))

    def test_local_publish_metadata_and_no_overwrite(self):
        run = self.root / "run-fixture"; run.mkdir()
        (run / "run_summary.json").write_text(json.dumps(dict(rehearsal=True,
            outcome_before_publish="test fixture only", exit_before_publish=0)), encoding="utf-8")
        (run / "payload.txt").write_text("fixture", encoding="utf-8")
        dest = self.root / "published"
        cmd = [sys.executable, "-B", str(WORK / "tools/u4_publish.py"),
               "--run", str(run), "--inte", str(dest), "--src", str(WORK)]
        first = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        manifests = list(dest.rglob("manifest.json")); self.assertEqual(len(manifests), 2)
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}
        for p in manifests:
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertTrue(d["complete"])
            self.assertEqual(d["retention"], "No retention policy is asserted by this tool; review generated data before sharing.")
            self.assertEqual(d["delete_rule"], "Retention and deletion are operator responsibilities; existing targets are not overwritten.")
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        self.assertEqual(second.returncode, 5)
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests})
        record = json.loads((run / "publish.json").read_text(encoding="utf-8"))
        self.assertTrue(record["adapter"]["reused"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
