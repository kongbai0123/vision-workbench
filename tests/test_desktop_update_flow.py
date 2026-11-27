"""One-click update ordering and recovery without Git mutations or a live window."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from workbench.desktop import MainWindow


class InlineThread:
    def __init__(self, *, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class UpdateWindow:
    def __init__(self):
        self.update_pending = self.closing = self.allow_close = False
        self.remote_update_result = self.remote_update_phase = self.remote_update_thread = None
        self.remote_update_started = 0.0
        self.remote_camera_result = None
        self.source_baseline = {}
        self.external_mode = self.editor_transition = None
        self.cvat_view = self.labelme_editor = None
        self.service = SimpleNamespace(
            data_root=Path("custom-project-data"), jobs=Mock(), training=Mock(), _cvat=None, _camera=None)
        self.service.jobs.active.return_value = False
        self.service.training.active_runs.return_value = []
        self.page = Mock()
        self.view = Mock()
        self.cvat_toolbar = Mock()
        self.stack = Mock()
        self.editor_status = Mock()
        self.transition_editor = Mock()


for method in ("apply_remote_update", "set_remote_update_status", "remote_update_busy",
               "prepare_remote_update", "poll_remote_update_save", "finish_remote_update_save",
               "finish_remote_update", "poll_remote_update", "finish_editor_sync",
               "stop_remote_update_camera", "poll_remote_update_camera", "start_remote_update"):
    setattr(UpdateWindow, method, getattr(MainWindow, method))


class DesktopUpdateFlowTests(unittest.TestCase):
    def setUp(self):
        self.window = UpdateWindow()
        self.pull = self.enterContext(patch("workbench.desktop.pull_remote_update", return_value={"state": "updated", "message": "pulled"}))
        self.restart = self.enterContext(patch("workbench.desktop.schedule_restart"))
        self.validate = self.enterContext(patch("workbench.desktop.validate_sources"))
        self.snapshot = self.enterContext(patch("workbench.desktop.source_snapshot", return_value={}))
        self.timer = self.enterContext(patch("workbench.desktop.QTimer.singleShot"))
        self.enterContext(patch("workbench.desktop.threading", SimpleNamespace(Thread=InlineThread)))
        self.enterContext(patch("workbench.desktop.QMessageBox.warning"))

    def begin_and_save(self):
        self.window.apply_remote_update()
        self.window.finish_remote_update_save(json.dumps({"state": "ready"}))

    def test_browser_save_precedes_git_and_editing_is_disabled_after_save(self):
        result = self.window.apply_remote_update()
        self.assertEqual(result["state"], "saving")
        script = "\n".join(call.args[0] for call in self.window.page.runJavaScript.call_args_list)
        self.assertIn("window.workbenchFlush()", script)
        self.assertIn("after.busy", script)
        self.assertIn("after.dirty", script)
        self.pull.assert_not_called()
        self.window.view.setEnabled.assert_not_called()
        self.window.finish_remote_update_save('{"state":"ready"}')
        self.pull.assert_called_once()
        self.window.view.setEnabled.assert_called_with(False)
        self.window.cvat_toolbar.setEnabled.assert_called_with(False)

    def test_tests_only_update_restarts_automatically_and_preserves_data_root(self):
        self.begin_and_save()
        self.window.poll_remote_update()
        self.validate.assert_called_once()
        self.restart.assert_called_once_with(self.validate.call_args.args[0], data_root=self.window.service.data_root)
        self.assertTrue(self.window.allow_close)
        self.assertEqual(self.window.remote_update_result["state"], "restarting")
        self.assertEqual(self.window.remote_update_phase, "restarting")
        self.assertTrue(self.window.update_pending)
        self.assertEqual(self.timer.call_args.args[0], 250)

    def test_active_jobs_training_or_cvat_are_queued_before_any_file_change(self):
        for kind in ("jobs", "training", "cvat"):
            with self.subTest(kind=kind):
                self.window = UpdateWindow()
                if kind == "jobs":
                    self.window.service.jobs.active.return_value = True
                elif kind == "training":
                    self.window.service.training.active_runs.return_value = ["run"]
                else:
                    self.window.service._cvat = Mock()
                    self.window.service._cvat.status.return_value = {"busy": True}
                self.assertEqual(self.window.apply_remote_update()["state"], "waiting")
                self.pull.assert_not_called()
                self.validate.assert_not_called()
                self.snapshot.assert_not_called()
                self.window.service.jobs.active.return_value = False
                self.window.service.training.active_runs.return_value = []
                if self.window.service._cvat:
                    self.window.service._cvat.status.return_value = {"busy": False}
                self.window.prepare_remote_update()
                self.assertEqual(self.window.remote_update_phase, "saving")

    def test_a_job_starting_during_save_defers_mutation_and_flushes_again(self):
        self.window.apply_remote_update()
        self.window.service.jobs.active.return_value = True
        self.window.finish_remote_update_save('{"state":"ready"}')
        self.assertEqual(self.window.remote_update_phase, "waiting")
        self.pull.assert_not_called()
        self.window.view.setEnabled.assert_not_called()
        self.window.service.jobs.active.return_value = False
        self.window.prepare_remote_update()
        scripts = [call.args[0] for call in self.window.page.runJavaScript.call_args_list]
        self.assertEqual(sum("window.workbenchFlush()" in script for script in scripts), 2)

    def test_external_editor_is_saved_through_existing_sync_before_browser_save(self):
        self.window.external_mode = "labelme"
        self.window.labelme_editor = Mock()
        self.window.apply_remote_update()
        self.window.transition_editor.assert_called_once_with("update")
        self.pull.assert_not_called()
        self.window.editor_transition = {"source": "labelme", "target": "update"}
        self.window.finish_editor_sync("")
        self.assertIsNone(self.window.external_mode)
        self.assertEqual(self.window.remote_update_phase, "saving")
        self.pull.assert_not_called()

    def test_external_sync_error_retains_original_editor_and_allows_retry(self):
        self.window.external_mode = "cvat"
        self.window.cvat_view = Mock()
        self.window.apply_remote_update()
        self.window.editor_transition = {"source": "cvat", "target": "update"}
        self.window.finish_editor_sync("annotation conflict")
        self.assertEqual(self.window.external_mode, "cvat")
        self.assertEqual(self.window.remote_update_result["state"], "blocked")
        self.assertFalse(self.window.update_pending)
        self.assertIsNone(self.window.remote_update_phase)
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_save_failure_never_pulls_and_retains_work_for_retry(self):
        self.window.apply_remote_update()
        self.window.finish_remote_update_save('{"state":"error","message":"save conflict"}')
        self.assertFalse(self.window.update_pending)
        self.assertEqual(self.window.remote_update_result["state"], "blocked")
        self.window.view.setEnabled.assert_called_with(True)
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.assertEqual(self.window.apply_remote_update()["state"], "saving")

    def test_duplicate_clicks_share_one_save_and_one_git_worker(self):
        self.window.apply_remote_update()
        self.window.apply_remote_update()
        scripts = [call.args[0] for call in self.window.page.runJavaScript.call_args_list]
        self.assertEqual(sum("window.workbenchFlush()" in script for script in scripts), 1)
        self.window.finish_remote_update_save('{"state":"ready"}')
        self.window.apply_remote_update()
        self.pull.assert_called_once()

    def test_background_exception_finishes_update_and_reenables_retry(self):
        self.pull.side_effect = RuntimeError("unexpected fetch failure")
        with self.assertLogs(level="ERROR"):
            self.begin_and_save()
        self.window.poll_remote_update()
        self.assertFalse(self.window.update_pending)
        self.assertEqual(self.window.remote_update_result["state"], "error")
        self.assertIn("unexpected fetch failure", self.window.remote_update_result["message"])
        self.window.view.setEnabled.assert_called_with(True)
        self.restart.assert_not_called()
        self.assertEqual(self.window.apply_remote_update()["state"], "saving")

    def test_already_current_does_not_restart_and_still_allows_later_checks(self):
        self.pull.return_value = {"state": "current", "message": "already current"}
        self.begin_and_save()
        self.window.poll_remote_update()
        self.restart.assert_not_called()
        self.assertFalse(self.window.update_pending)
        self.assertEqual(self.window.remote_update_result["state"], "current")
        self.assertEqual(self.window.apply_remote_update()["state"], "saving")

    def test_restart_failure_retains_backup_metadata_and_does_not_close(self):
        self.pull.return_value = {"state": "updated", "message": "pulled", "backup_ref": "backup-sha", "backup_path": "recovery/source"}
        self.restart.side_effect = OSError("could not create restart helper")
        self.begin_and_save()
        with self.assertLogs(level="ERROR"):
            self.window.poll_remote_update()
        self.assertFalse(self.window.allow_close)
        self.assertFalse(self.window.update_pending)
        self.assertEqual(self.window.remote_update_result["backup_ref"], "backup-sha")
        self.assertEqual(self.window.remote_update_result["backup_path"], "recovery/source")
        self.assertEqual(self.window.remote_update_result["state"], "error")
        self.window.view.setEnabled.assert_called_with(True)

    def test_live_preview_is_stopped_and_checked_before_git_or_restart(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "running", "running": True, "recording": False}
        camera.stop.return_value = {"state": "stopped", "running": False, "recording": False, "error": None}
        self.begin_and_save()
        self.assertEqual(self.window.remote_update_phase, "camera")
        camera.stop.assert_called_once_with()
        camera.stop_recording.assert_not_called()
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.assertFalse(self.window.allow_close)
        self.window.poll_remote_update_camera()
        self.pull.assert_called_once()
        self.window.poll_remote_update()
        self.restart.assert_called_once()

    def test_recording_is_finalized_before_camera_stop_and_before_git(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "running", "running": True, "recording": True}
        camera.stop_recording.return_value = {"recording": False, "error": None,
                                              "last_recording": {"path": "saved-video.mp4", "frames": 8}}
        camera.stop.return_value = {"state": "stopped", "recording": False, "running": False, "error": None}
        sequence = Mock()
        sequence.attach_mock(camera.stop_recording, "finish_recording")
        sequence.attach_mock(camera.stop, "stop_camera")
        sequence.attach_mock(self.pull, "pull")
        self.begin_and_save()
        self.assertEqual([call[0] for call in sequence.mock_calls], ["finish_recording", "stop_camera"])
        self.assertFalse(self.window.allow_close)
        self.window.poll_remote_update_camera()
        self.assertEqual([call[0] for call in sequence.mock_calls], ["finish_recording", "stop_camera", "pull"])

    def test_unfinished_recording_prevents_camera_shutdown_git_and_restart(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "running", "recording": True}
        camera.stop_recording.return_value = {"recording": False, "last_recording": None}
        with self.assertLogs(level="ERROR"):
            self.begin_and_save()
            self.window.poll_remote_update_camera()
        camera.stop.assert_not_called()
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.assertFalse(self.window.allow_close)
        self.assertFalse(self.window.update_pending)
        self.assertEqual(self.window.remote_update_result["state"], "blocked")
        self.window.view.setEnabled.assert_called_with(True)

    def test_driver_still_stopping_is_polled_without_git_or_allow_close(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "stopping", "running": True, "recording": False}
        camera.stop.return_value = {"state": "stopping", "recording": False, "error": "waiting for driver"}
        self.begin_and_save()
        self.window.poll_remote_update_camera()
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.assertFalse(self.window.allow_close)
        camera.status.return_value = {"state": "stopped", "running": False, "recording": False, "error": None}
        self.window.poll_remote_update_camera()
        self.window.poll_remote_update_camera()
        self.pull.assert_called_once()

    def test_camera_stop_error_retains_window_and_never_starts_git(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "starting", "recording": False}
        camera.stop.side_effect = RuntimeError("camera driver failed")
        with self.assertLogs(level="ERROR"):
            self.begin_and_save()
            self.window.poll_remote_update_camera()
        self.pull.assert_not_called()
        self.restart.assert_not_called()
        self.assertFalse(self.window.allow_close)
        self.assertFalse(self.window.update_pending)
        self.assertIn("camera driver failed", self.window.remote_update_result["message"])

    def test_driver_stop_timeout_preserves_window_and_retry_without_git(self):
        camera = self.window.service._camera = Mock()
        camera.status.return_value = {"state": "stopping", "recording": True}
        camera.stop_recording.return_value = {"recording": False, "error": None,
                                              "last_recording": {"path": "saved-video.mp4", "frames": 8}}
        camera.stop.return_value = {"state": "stopping", "recording": True}
        self.begin_and_save()
        self.window.remote_update_started -= 31
        with self.assertLogs(level="ERROR"):
            self.window.poll_remote_update_camera()
        self.assertFalse(self.window.allow_close)
        self.assertFalse(self.window.update_pending)
        self.window.view.setEnabled.assert_called_with(True)
        self.pull.assert_not_called()
        self.restart.assert_not_called()

    def test_native_close_cannot_interrupt_update_transaction(self):
        for phase in ("waiting", "saving", "camera", "updating"):
            with self.subTest(phase=phase):
                self.window.update_pending = True
                self.window.remote_update_phase = phase
                event = Mock()
                MainWindow.closeEvent(self.window, event)
                event.ignore.assert_called_once_with()
                event.accept.assert_not_called()
        self.window.page.runJavaScript.assert_not_called()
        self.window.allow_close = True
        event = Mock()
        MainWindow.closeEvent(self.window, event)
        event.accept.assert_called_once_with()
        event.ignore.assert_not_called()

    @unittest.skipUnless(shutil.which("node"), "Node.js browser callback test")
    def test_browser_promise_does_not_mark_failed_or_busy_saves_ready(self):
        self.window.apply_remote_update()
        script = next(call.args[0] for call in self.window.page.runJavaScript.call_args_list
                      if "window.workbenchFlush()" in call.args[0])
        source = """
            import assert from 'node:assert/strict';
            const saveScript=SCRIPT;
            for(const [save,busy,expected] of [[false,false,'error'],[true,true,'waiting'],[true,false,'ready']]){
                globalThis.window={workbenchState:()=>({busy,dirty:false}),workbenchFlush:async()=>save};
                eval(saveScript);
                await new Promise(resolve=>setTimeout(resolve,0));
                assert.equal(window.__workbenchRemoteUpdate.state,expected);
            }
            let autoCapture=true,flushed=false;
            globalThis.window={workbenchState:()=>({busy:autoCapture,dirty:false}),
                workbenchFlush:async()=>{flushed=true;autoCapture=false;return true}};
            eval(saveScript);
            await new Promise(resolve=>setTimeout(resolve,0));
            assert.equal(flushed,true);
            assert.equal(window.__workbenchRemoteUpdate.state,'ready');
        """.replace("SCRIPT", json.dumps(script))
        subprocess.run([shutil.which("node"), "--input-type=module", "-"], input=source,
                       text=True, encoding="utf-8", capture_output=True, timeout=10, check=True)


class SettingsUpdateFlowTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js settings state test")
    def test_update_button_tracks_restart_without_manual_step_and_allows_error_retry(self):
        module = (Path(__file__).resolve().parents[1] / "web/pages/settings.mjs").as_uri()
        source = """
            import assert from 'node:assert/strict';
            import {createSettingsPage} from MODULE;
            globalThis.document={querySelectorAll:()=>[]};
            const elements=new Map();
            const $=id=>{if(!elements.has(id))elements.set(id,{textContent:'',className:'',disabled:false});return elements.get(id)};
            const callbacks={};let calls=0,complete;
            const state={nativeBridge:{applyRemoteUpdate:callback=>{calls++;complete=callback}}};
            const page=createSettingsPage({$,state,nativeCallbacks:callbacks,toast:()=>{},setUpdateIndicators:()=>{}});
            const first=page.runRemoteUpdate();
            await page.runRemoteUpdate();assert.equal(calls,1);
            complete(JSON.stringify({state:'saving',message:'saving'}));await first;
            assert.equal($('githubUpdateButton').disabled,true);
            callbacks.remoteUpdateStatus({state:'restarting',message:'automatic restart'});
            assert.match($('githubUpdateButton').textContent,/正在重新啟動/);
            assert.doesNotMatch($('githubUpdateButton').textContent,/請重新啟動/);
            page.renderRemoteVersion({state:'current',message:'old background version check'});
            assert.equal($('githubVersionStatus').textContent,'automatic restart');
            callbacks.remoteUpdateStatus({state:'error',message:'retry available'});
            assert.equal($('githubUpdateButton').disabled,false);
            const retry=page.runRemoteUpdate();assert.equal(calls,2);
            complete(JSON.stringify({state:'current',message:'already current'}));await retry;
            assert.equal($('githubUpdateButton').disabled,false);
        """.replace("MODULE", json.dumps(module))
        subprocess.run([shutil.which("node"), "--input-type=module", "-"], input=source,
                       text=True, encoding="utf-8", capture_output=True, timeout=10, check=True)


if __name__ == "__main__":
    unittest.main()
