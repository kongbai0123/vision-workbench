import tempfile
import unittest
import os
import subprocess
import sys
import time
from pathlib import Path

from workbench.desktop_update import changed_sources, source_snapshot, validate_sources


class DesktopUpdateTests(unittest.TestCase):
    def test_detects_program_changes_and_ignores_project_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"workbench").mkdir();(root/"data").mkdir()
            (root/"main.py").write_text("value=1",encoding="utf-8")
            (root/"workbench"/"unit.py").write_text("value=1",encoding="utf-8")
            (root/"data"/"project.sqlite3").write_bytes(b"first")
            before=source_snapshot(root)
            (root/"workbench"/"unit.py").write_text("value=2",encoding="utf-8")
            (root/"data"/"project.sqlite3").write_bytes(b"second")
            self.assertEqual(changed_sources(before,source_snapshot(root)),["workbench/unit.py"])

    def test_invalid_python_blocks_restart_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"workbench").mkdir()
            (root/"workbench"/"broken.py").write_text("def broken(",encoding="utf-8")
            with self.assertRaises(SyntaxError):validate_sources(root)

    @unittest.skipUnless(os.name=="nt","Windows restart handoff")
    def test_restart_helper_waits_for_current_process(self):
        import workbench.desktop_update as updater
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"main.py").write_text("from pathlib import Path\nPath(__file__).with_name('started').write_text('ok')",encoding="utf-8")
            parent=subprocess.Popen([sys.executable,"-c","import time;time.sleep(.5)"])
            helper=subprocess.Popen([sys.executable,str(Path(updater.__file__)),"--wait",str(parent.pid),str(root)])
            try:
                time.sleep(.1);self.assertFalse((root/"started").exists())
                parent.wait(timeout=5);self.assertEqual(helper.wait(timeout=5),0)
                deadline=time.monotonic()+5
                while not (root/"started").exists() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue((root/"started").exists())
            finally:
                for process in (parent,helper):
                    if process.poll() is None:process.terminate()


if __name__=="__main__":unittest.main()
