import tempfile
import unittest
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workbench.desktop_update import changed_sources, source_snapshot, validate_sources, missing_runtime_requirements


class DesktopUpdateTests(unittest.TestCase):
    def test_in_page_update_status_and_start_do_not_open_a_dialog(self):
        from workbench.desktop import MainWindow

        class FakeWindow:
            update_pending=False
            closing=False
            pending_source_changes=[]
            external_mode=None
            service=SimpleNamespace(
                jobs=SimpleNamespace(active=lambda:False),
                training=SimpleNamespace(active_runs=lambda:False),
                _cvat=None,
            )
            page=SimpleNamespace(scripts=[],runJavaScript=lambda script,*args:FakeWindow.page.scripts.append(script))

            def check_update_indicator(self):
                self.pending_source_changes=["web/app.mjs","web/index.html"]
                return list(self.pending_source_changes)

            def desktop_update_status(self):
                return MainWindow.desktop_update_status(self)

            def notify_update_progress(self,state,message):
                self.progress=(state,message)

            def poll_update(self):
                pass

        window=FakeWindow()
        status=window.desktop_update_status()
        self.assertEqual(status["state"],"available")
        self.assertEqual(status["count"],2)
        with patch("workbench.desktop.QTimer.singleShot") as timer:
            result=MainWindow.apply_update(window)
        self.assertEqual(result["state"],"updating")
        self.assertTrue(window.update_pending)
        self.assertEqual(window.progress[0],"saving")
        self.assertIn("window.workbenchFlush()",window.page.scripts[-1])
        timer.assert_called_once()

    def test_already_installed_requirements_allow_restart(self):
        from importlib.metadata import version
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'requirements.txt'
            path.write_text('PySide6=='+version('PySide6')+'\n','utf-8')
            self.assertEqual(missing_runtime_requirements(folder),[])
            path.write_text('PySide6==0.0.0\nworkbench-missing-fixture==1.0\n','utf-8')
            self.assertEqual(len(missing_runtime_requirements(folder)),2)

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
