"""Opt-in native desktop smoke using the installed offline MM and Qwen runtime."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hammer", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-checks", action="store_true", help="Include SAM boundary and Gemma observation checks")
    parser.add_argument("--show", action="store_true", help="Keep the checked project open in one desktop window")
    args = parser.parse_args()
    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding")

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from workbench.desktop import DialogBridge, MainWindow
    from workbench.server import WorkbenchService

    root = args.output.resolve()
    root.mkdir(parents=True)
    paths = [root / f"{name}{source.suffix.lower()}" for name, source in
             [("01-hammer", args.hammer), ("02-review", args.second)]]
    for original, copy in zip((args.hammer, args.second), paths):
        shutil.copyfile(original, copy)
    app = QApplication([])
    app.setApplicationName("Vision Workbench")
    bridge = DialogBridge()
    service = WorkbenchService(root / "data", dialog=bridge).start()
    window = None
    try:
        project_name = "AI 框界與裁決示範" if args.model_checks else "MM ＋ Qwen · 鐵鎚與例外覆核示範"
        project_id = service.store.create_project(project_name)["id"]
        project = service.store.get_project(project_id)
        service.store.manage_classes(project_id, ["hammer", "screwdriver", "wrench"], {}, project["revision"])
        asset_ids = service.store.add_assets(project_id, [{"path": path, "shapes": []} for path in paths])["asset_ids"]
        hammer_id = asset_ids[0]
        window = MainWindow(service, bridge)
        window.resize(1440, 900)
        window.show()
        summary = exercise_ui(window, service, project_id, hammer_id, asset_ids[1], root,
                              (QEventLoop, QPoint, Qt, QTimer, QTest), model_checks=args.model_checks)
        (root / "smoke-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.show:
            print("READY: desktop demonstration remains open", flush=True)
            return app.exec()
        return 0
    finally:
        if window is not None:
            window.allow_close = True
            window.close()
        service.close()


def exercise_ui(window, service, pid, hammer_id, exception_id, root, qt, *, model_checks=False):
    QEventLoop, QPoint, Qt, QTimer, QTest = qt

    def js(script):
        result = []
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        window.page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
        timer.start(10000)
        if not result:
            loop.exec()
        timer.stop()
        if not result:
            raise AssertionError("Qt JavaScript callback timed out")
        return result[0]

    def wait(expression, seconds=25):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if js(f"Boolean({expression})"):
                return
            QTest.qWait(150)
        raise AssertionError(expression + "\n" + js("document.body.innerText.slice(-3500)"))

    def click(selector):
        selector = json.dumps(selector)
        wait(f"document.querySelector({selector}) && !document.querySelector({selector}).disabled")
        js(f"document.querySelector({selector}).click()")

    def choose(selector, value):
        js(f"{{const el=document.querySelector({json.dumps(selector)});el.value={json.dumps(value)};"
           "el.dispatchEvent(new Event('change',{bubbles:true}));}")

    def save():
        js("document.querySelector('#saveNow').click()")
        wait("!window.workbenchState().dirty && !window.workbenchState().saving")

    def open_assist():
        click('#annotationAiTab')
        wait("!document.querySelector('#annotationAiPanel').hidden")
        click('#annotationBatchTab')
        wait("!document.querySelector('#annotationAssistPanel').hidden")

    wait("document.querySelector('.project-card-open')")
    click(".project-card-open")
    wait(f"window.workbenchState?.().projectId === {json.dumps(pid)}")
    click('[data-stage="annotate"]')
    wait("window.workbenchState().stage === 'annotate' && document.querySelector('#assetImage').complete")
    open_assist()
    choose("#annotationAssistScope", "visible")
    for selector in ('#annotationAssistEnableSam', '#annotationAssistEnableGemma'):
        assert js(f"document.querySelector({json.dumps(selector)}).checked") is True
        if not model_checks:
            click(selector)
    assert js("document.querySelector('#annotationAssistPanel .annotation-assist-checks label:last-child input').checked") is False
    started = time.monotonic()
    click("#annotationAssistStart")
    wait("document.querySelector('#annotate').dataset.messageSource === 'batch' && "
         "!document.querySelector('#annotationMessagesPanel').hidden")
    wait("document.querySelector('button[data-assist-candidate-id]') || "
         "document.querySelector('#annotationAssistStatus')?.textContent.includes('失敗')", 600)
    wait("document.querySelector('button[data-assist-candidate-id]')")
    assert service.store.get_asset(pid, hammer_id)["shapes"] == [], "AI silently wrote formal labels"
    batch_id = js("document.querySelector('#annotationAssistRuns').value")
    initial = service.annotation_assist.detail(pid, batch_id)
    assert initial['model_checks'] == {'sam': model_checks, 'gemma': model_checks}
    assert initial["isolation"]["network_denied"] and initial["isolation"]["cuda"]
    record = next((root / "data" / "projects").glob(f"*/integrations/annotation-assist/{batch_id}/batch.json"))
    candidate_digest = sha256(record.read_bytes()).hexdigest()
    if model_checks:
        assert initial.get('gemma', {}).get('status') == 'observed', initial.get('gemma')
        assert initial['gemma']['observation_only'] and initial['gemma']['observed_candidates'] >= 2
        proof = json.loads((record.parent / 'inference-gemma-isolation.json').read_bytes())
        assert proof['appcontainer_token'] and proof['network_denied'] and proof['os_network_capabilities'] == 0
        assert any(row.get('sam', {}).get('mask_box_xyxy') for frame in initial['frames'] for row in frame['candidates'])
    print("PASS: real batch completed and proposals stayed separate", flush=True)

    if js("window.workbenchState().assetId") != hammer_id:
        click(f'#annotationMessageBatchFrames [data-asset-id="{hammer_id}"]')
        wait(f"window.workbenchState().assetId === {json.dumps(hammer_id)}")
    manual_label = js("document.querySelector('[data-assist-manual-label]')?.dataset.assistManualLabel")
    assert manual_label, "an unresolved class did not request human boxing"
    click(f'[data-assist-manual-label="{manual_label}"]')
    wait("document.querySelector('#annotationMessagesPanel').hidden && "
         "document.querySelector('#addBoundingBox').getAttribute('aria-pressed') === 'true'")
    assert js("document.querySelector('#shapeLabel').value") == manual_label
    assert js("window.workbenchState().assetId") == hammer_id
    click('#openAnnotationMessages')
    click('#annotationBatchMessagesTab')
    candidate = 'button[data-assist-label="hammer"][data-assist-verdict="accepted"]'
    wait(f"document.querySelector({json.dumps(candidate)})")
    click(candidate)
    if model_checks:
        wait("document.querySelector('#annotationAssistCandidateModels').textContent.includes('Gemma 4')")
        wait("document.querySelector('#overlay [data-assist-preview]') && document.querySelector('#overlay [data-assist-sam-preview]')")
        click('#annotationAssistShowOriginal')
        wait("!document.querySelector('#overlay [data-assist-preview]') && document.querySelector('#overlay [data-assist-sam-preview]')")
        click('#annotationAssistShowOriginal')
        click('#annotationAssistShowSam')
        wait("document.querySelector('#overlay [data-assist-preview]') && !document.querySelector('#overlay [data-assist-sam-preview]')")
        click('#annotationAssistShowSam')
    QTest.qWait(200)
    window.grab().save(str(root / "01-model-evidence.png"))
    click("#annotationAssistAdopt")
    wait("document.querySelector('.shape-row')")
    save()
    first = service.store.get_asset(pid, hammer_id)
    assert len(first["shapes"]) == 1
    origin = first['shapes'][0]['metadata']
    assert origin['source'] == 'manual' and origin['annotation_assist']['human_confirmed']
    archived = f'button[data-assist-candidate-id="{origin["annotation_assist"]["candidate_id"]}"]'
    assert js(f"Boolean(document.querySelector('#annotationAssistArchivedCandidates ' + {json.dumps(archived)}))")
    assert not js(f"Boolean(document.querySelector('#annotationAssistCandidates ' + {json.dumps(archived)}))")
    click('#returnAnnotationEditor')
    wait("document.querySelector('#annotationMessagesPanel').hidden")
    click('#annotationObjectsTab')
    click('#shapeList .shape-row')
    click('#editBoundingBox')
    coords = json.loads(js("""JSON.stringify((()=>{
        const h=document.querySelector('#overlay [data-handle="0"]').getBoundingClientRect();
        return [h.left+h.width/2,h.top+h.height/2];
    })())"""))
    start = QPoint(round(coords[0]), round(coords[1]))
    end = start + QPoint(-12, -8)
    target = window.view.focusProxy() or window.view
    button, modifier = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mousePress(target, button, modifier, target.mapFrom(window.view, start))
    QTest.mouseMove(target, target.mapFrom(window.view, end), 80)
    QTest.mouseRelease(target, button, modifier, target.mapFrom(window.view, end))
    save()
    corrected = service.store.get_asset(pid, hammer_id)
    assert len(corrected["shapes"]) == 1 and corrected["revision"] > first["revision"]
    shape = corrected["shapes"][0]
    assert shape["label"] == "hammer" and shape["x"] < first["shapes"][0]["x"]
    window.grab().save(str(root / "02-saved-correction.png"))
    print("PASS: dragged a native SVG box handle, saved geometry, kept the original candidate", flush=True)

    click('#openAnnotationMessages')
    click('#annotationBatchMessagesTab')
    click("#annotationAssistResolve")
    wait("document.querySelector('#annotationAssistExceptions').textContent.includes('已完成人工覆核')")
    click('[data-stage="review"]')
    wait("window.workbenchState().stage === 'review' && document.querySelector('.review-card')")
    window.grab().save(str(root / "03-review.png"))
    wait("document.querySelector('#annotationAssistReviewLink').getClientRects().length > 0")
    click("#annotationAssistReviewLink")
    wait(f"window.workbenchState().stage === 'annotate' && window.workbenchState().assetId === {json.dumps(exception_id)}")
    wait("document.querySelector('#annotate').dataset.messageSource === 'batch'")
    wait("document.querySelector('#annotationAssistExceptions').textContent.includes('待人工覆核')")
    assert service.store.get_asset(pid, exception_id)["shapes"] == []
    QTest.qWait(200)
    window.grab().save(str(root / "04-exception.png"))
    click(f'#annotationMessageBatchFrames [data-asset-id="{hammer_id}"]')
    wait(f"window.workbenchState().stage === 'annotate' && window.workbenchState().assetId === {json.dumps(hammer_id)}")
    click('#returnAnnotationEditor')
    wait(f"document.querySelector('[data-shape={json.dumps(shape['id'])}]')")
    reloaded = service.store.get_asset(pid, hammer_id)
    assert reloaded["shapes"] == corrected["shapes"]
    assert sha256(record.read_bytes()).hexdigest() == candidate_digest
    window.grab().save(str(root / "05-annotation-workbench.png"))
    print("PASS: a single window preserved the saved box through the review and editor stages", flush=True)
    return {"project_id": pid, "asset_id": hammer_id, "batch_id": batch_id, "candidate_sha256": candidate_digest,
            "batch_summary": initial["summary"], "inference_seconds": initial["timing"]["inference_seconds"],
            "sam": initial.get('sam'), "gemma": initial.get('gemma'),
            "before": first["shapes"][0], "after": shape, "saved_revision": reloaded["revision"],
            "elapsed_seconds": round(time.monotonic() - started, 2)}


if __name__ == "__main__":
    raise SystemExit(main())
