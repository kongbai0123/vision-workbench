"""Exercise training monitoring through real Qt Chromium and persisted run reports.

Run explicitly: .venv/Scripts/python.exe tests/desktop_training.py
The isolated dataset and metric files are fixtures; no training worker is started.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService
from workbench.training_engine import atomic_json
from composer_core.geometry import encode_rle


class Page(QWebEnginePage):
    def __init__(self, *args):
        super().__init__(*args)
        self.errors = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.errors.append(f"{source}:{line}: {message}")


def seed_reports(service, folder):
    """Use the real store/snapshot boundary, then seed saved worker output."""
    project = service.store.create_project("訓練監控與疊圖驗收")
    pid = project["id"]
    assets = []
    for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
        source = folder / f"sample-{index}.png"
        Image.new("RGB", (48, 36), (30 + index * 20, 50, 80)).save(source)
        shape = {"id": f"part-{index}", "type": "rectangle", "label": "工件" if index % 2 else "配件",
                 "x": 8, "y": 6, "width": 25, "height": 24}
        if index == 0:
            mask = np.zeros((36, 48), np.uint8); mask[6:30, 8:33] = 1
            mask[15, 20] = 0; mask[18, 22:25] = 0
            shape = {"id": f"part-{index}", "type": "mask", "label": "配件", "x": 0, "y": 0,
                     "width": 48, "height": 36, "counts": encode_rle(mask)}
        assets.append({"path": str(source), "name": source.name, "split": split,
                       "batch_id": f"{split}-{index}", "review_state": "approved",
                       "source": {"kind": "test"}, "shapes": [shape]})
    service.store.add_assets(pid, assets)
    snapshot = service.training.create_dataset_version(pid)
    assert snapshot["splits"] == {"train": 2, "val": 2, "test": 2}, snapshot
    saved = {}
    base = time.time() - 100
    for number in (1, 2, 3, 4):
        run_id = f"R{number:03d}"
        classification = number == 3
        count = 12 if number == 4 else 7 if number == 2 else 10
        rows = []
        for epoch in range(1, count + 1):
            row = {"epoch": epoch, "train/loss": round(1.7 / epoch + number * .01, 4)}
            if classification:
                row.update({"val/accuracy": round(.6 + epoch * .02, 4),
                            "val/macro_f1": round(.55 + epoch * .025, 4),
                            "val/macro_recall": round(.5 + epoch * .03, 4)})
            elif number == 2 and epoch == 3:
                row["val/mean_iou"] = None
            elif number != 2 or epoch != 4:
                row["val/mean_iou"] = round(.5 + epoch * .035 + number * .01, 4)
            rows.append(row)
        score = ({"accuracy": .81, "macro_f1": .80, "macro_recall": .78,
                  "confusion_matrix": [[1, 0], [0, 1]], "classes": ["配件", "工件"]}
                 if classification else {"mean_iou": .86 - number * .01,
                                         "per_class_iou": {"配件": .81, "工件": .87}})
        run = {"schema_version": 1, "run_id": run_id, "project_id": pid,
               "dataset_version_id": snapshot["id"], "model_version_id": f"M{number:03d}",
               "engine": "resnet18_classification" if classification else "maskrcnn_resnet50_fpn",
               "engine_name": "ResNet18 · 分類" if classification else "Mask R-CNN · ResNet50 FPN",
               "config": {"epochs": 20 if number == 4 else 10, "device": "auto", "seed": 42 + number,
                          "image_size": 224 if classification else 640, "batch_size": 1,
                          "learning_rate": .0005}, "device": "cpu", "status": "completed",
               "message": "訓練與評估完成", "epoch": count, "progress": 100,
               "created_at": base + (-10 if number == 4 else 0 if classification else number),
               "updated_at": base + 20, "completed_at": base + 20,
               "metrics": rows[-1], "evaluation": {
                   "validation": {"split": "val", "images": 2, **score},
                   "test": {"split": "test", "images": 2, **score}}}
        if number == 4:
            run.update(status="stopped", progress=60, message="Epoch 12 / 20")
            run.pop("evaluation")
            run.pop("completed_at")
        path = service.training._run_path(pid, run_id)
        path.parent.mkdir(parents=True)
        atomic_json(path, run)
        (path.parent / "metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        if number < 4:
            model = service.training._model_path(pid, run["model_version_id"])
            model.parent.mkdir(parents=True)
            atomic_json(model, {"schema_version": 1, "model_version_id": run["model_version_id"],
                               "run_id": run_id, "project_id": pid, "dataset_version_id": snapshot["id"],
                               "engine": run["engine"], "engine_name": run["engine_name"],
                               "task": "image_classification" if classification else "instance_segmentation",
                               "created_at": base + number, "classes": snapshot["classes"],
                               "evaluation": run["evaluation"]})
        saved[run_id] = (path, run, rows)
    return pid, saved


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "desktop"
    output.mkdir(parents=True, exist_ok=True)
    steps = []
    with tempfile.TemporaryDirectory(prefix="vision-training-ui-") as directory:
        folder = Path(directory)
        service = WorkbenchService(folder / "data").start()
        pid, saved = seed_reports(service, folder)
        view = QWebEngineView()
        page = Page(view)
        view.setPage(page)
        view.resize(1440, 1300)
        view.show()
        view.setUrl(QUrl(service.url))
        page.setVisible(True)

        def js(script):
            result = []
            loop = QEventLoop()
            page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            assert result, "JavaScript timed out"
            return result[0]

        def wait(expression, seconds=30):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(expression):
                    return
                QTest.qWait(80)
            raise AssertionError("Timeout: " + expression + "\n" + str(js("document.body.innerText.slice(-4000)")))

        def click(selector):
            encoded = json.dumps(selector)
            wait(f"(()=>{{const e=document.querySelector({encoded});return e&&!e.disabled}})()")
            js(f"document.querySelector({encoded}).click()")

        def fill(selector, value):
            js(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(str(value))};e.dispatchEvent(new Event('input',{{bubbles:true}}));e.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")

        def structured(script):
            return json.loads(js(f"JSON.stringify({script})"))

        def set_model(run_id, checked):
            selector = f'#trainingModelPicker input[data-model-run="{run_id}"]'
            if js(f"document.querySelector({json.dumps(selector)}).checked") != checked:
                click(selector)

        def plotted_metrics():
            return structured("[...document.querySelectorAll('#trainingPlots [data-metric-key]')].map(e=>e.dataset.metricKey)")

        def epoch_stats():
            return structured("(()=>{const e=document.querySelector('.training-epoch-scroll'),t=e.querySelector('table'), rows=[...t.tBodies[0].rows];return {height:e.clientHeight,total:e.scrollHeight,scroll:e.scrollTop,rows:rows.length,rowHeight:rows[0]?.getBoundingClientRect().height||0,header:t.tHead.getBoundingClientRect().height,headerTop:t.tHead.getBoundingClientRect().top,viewportTop:e.getBoundingClientRect().top,sticky:getComputedStyle(t.tHead).position==='sticky'||[...t.tHead.querySelectorAll('th')].every(h=>getComputedStyle(h).position==='sticky')}})()")

        def capture(name, monitor=False, target=None):
            target = target or ("#trainingRunDetail" if monitor else None)
            if target:
                js(f"document.querySelector({json.dumps(target)}).scrollIntoView({{block:'start'}})")
            # Offscreen Chromium can retain a compositor frame after a dialog closes.
            # A resize invalidates the surface so the PNG represents the verified DOM.
            width, height = view.width(), view.height()
            view.resize(width + 1, height)
            QTest.qWait(80)
            view.resize(width, height)
            QTest.qWait(500)
            view.grab()  # Request a compositor readback before the saved frame.
            QTest.qWait(500)
            view.grab().save(str(output / f"{name}.png"))
            assert not js("document.documentElement.scrollWidth>innerWidth+2"), name
            steps.append(name)
            print("PASS", name, flush=True)

        def history():
            click("#openTrainingHistory")
            wait("document.querySelector('#trainingHistoryDialog').open")

        def chart_script(metric, suffix=""):
            return "document.querySelector(" + json.dumps(f'#trainingPlots [data-metric-key="{metric}"]') + ")" + suffix

        def domains():
            return json.loads(js("JSON.stringify([...document.querySelectorAll('#trainingPlots svg')].map(s=>[s.dataset.xMax,s.dataset.yMax]))"))

        def hover_epoch(metric, epoch):
            script = chart_script(metric)
            js(f"""(()=>{{const card={script}, hit=card.querySelector('[data-chart-hit]'),
                rect=hit.getBoundingClientRect(), max=Number(card.querySelector('svg').dataset.xMax);
                hit.dispatchEvent(new PointerEvent('pointermove',{{bubbles:true,
                    clientX:rect.left+rect.width*({epoch}-1)/(max-1),clientY:rect.top+rect.height/2}}));}})()""")
            return js(script + ".querySelector('.training-chart-hover').textContent")

        try:
            wait("typeof window.workbenchState==='function'")
            click(".project-card-open")
            wait("!document.querySelector('[data-stage=train]').disabled")
            click("[data-stage=split]")
            wait("document.querySelector('#splitFlowStats').innerText.includes('獨立來源群組')")
            capture("29-data-split-stage", target="#split")
            click("#continueToTraining")
            wait("window.workbenchState().stage==='train'&&!window.workbenchState().transitioning")
            wait("document.querySelectorAll('#trainingPlots svg').length===2")
            assert js("document.querySelector('#trainingRunDetail').innerText.includes('R002')")
            # The split manager stays accessible even when readiness is already true.
            old_manifest = service.training.datasets / pid / "D001" / "manifest.json"
            old_bytes = old_manifest.read_bytes()
            click("#prepareAutoSplit")
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            click("#openSplitFlowManager")
            wait("document.querySelectorAll('#smartSplitDialog tbody tr').length===6")
            click("#previewSmartSplit")
            wait("!document.querySelector('#applySmartSplitVersion').disabled")
            capture("30-smart-split-preview")
            fill("#smartSplitSeed",43)
            assert js("document.querySelector('#applySmartSplit').disabled")
            click("#previewSmartSplit")
            wait("!document.querySelector('#applySmartSplitVersion').disabled")
            click("#applySmartSplitVersion")
            wait("!document.querySelector('#smartSplitDialog').open")
            wait("document.querySelector('#trainingDataset').value==='D002'")
            assert old_manifest.read_bytes() == old_bytes
            assert service.store.get_project(pid)['split_plan']['current']
            # YOLO Seg compatibility is visible during review and checks current annotations.
            click("[data-stage=review]")
            wait("window.workbenchState().stage==='review'&&!window.workbenchState().transitioning")
            wait("document.querySelector('#yoloCompatibilityBadge').textContent==='需要處理'")
            capture("32-yolo-compatibility-review", target="#yoloCompatibilityPanel")
            click("#yoloCompatibilityReport .yolo-issue .text-button")
            wait("document.querySelector('#formDialog').open&&document.querySelector('.yolo-location-canvas')?.dataset.ready==='true'")
            assert js("document.querySelectorAll('.yolo-hole-card').length") == 2
            click(".yolo-hole-card:nth-child(2)")
            wait("document.querySelector('.yolo-hole-counter').textContent==='2 / 2'&&document.querySelector('.yolo-location-canvas').dataset.ready==='true'")
            capture("33-yolo-hole-location", target="#formDialog")
            click("#cancelDialog")
            click("[data-stage=split]")
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            click("#continueToTraining")
            wait("window.workbenchState().stage==='train'&&!window.workbenchState().transitioning")
            # Parameter forms expose the selected engine's actual supported schema.
            fill("#trainingEngine", "maskrcnn_resnet50_fpn")
            wait("!!document.querySelector('#trainingAdvancedFields [data-training-param=learning_rate]')")
            values = {"image_size": "256", "batch_size": "2", "learning_rate": "0.003",
                      "weight_decay": "0.0002", "optimizer": "SGD"}
            assert structured("[...document.querySelectorAll('#trainingAdvancedFields [data-training-param]')].map(e=>e.dataset.trainingParam).sort()") == sorted([*values,"scheduler","min_learning_rate","warmup_epochs","lr_patience","lr_factor"])
            for key, value in values.items():
                fill(f'#trainingAdvancedFields [data-training-param="{key}"]', value)
            fill("#trainingEngine", "pixel_prototype_v1")
            wait("!!document.querySelector('#trainingAdvancedFields [data-training-param=threshold_min]')")
            assert structured("[...document.querySelectorAll('#trainingAdvancedFields [data-training-param]')].map(e=>e.dataset.trainingParam).sort()") == ["threshold_max", "threshold_min"]
            assert js("document.querySelector('#trainingSeed').getClientRects().length===0&&document.querySelector('#trainingDevice').getClientRects().length===0")
            fill('#trainingAdvancedFields [data-training-param="threshold_min"]', "1")
            fill('#trainingAdvancedFields [data-training-param="threshold_max"]', "2")
            fill("#trainingEngine", "maskrcnn_resnet50_fpn")
            wait("!!document.querySelector('#trainingAdvancedFields [data-training-param=learning_rate]')")
            for key, value in values.items():
                assert js(f"document.querySelector('#trainingAdvancedFields [data-training-param={key}]').value") == value
            # YOLO Seg keeps conversion policy controls; the fixed version is rechecked at Run start.
            fill("#trainingEngine", "yolo26n_seg")
            assert js("document.querySelector('#trainingParam-yolo_mask_policy').value") == "repair_tiny_holes"
            click("#trainingCompatibilitySettings summary")
            fill("#trainingParam-yolo_mask_policy", "strict")
            assert js("document.querySelector('#trainingParam-tiny_hole_max_pixels').getClientRects().length===0")
            fill("#trainingParam-yolo_mask_policy", "repair_tiny_holes")
            fill("#trainingParam-tiny_hole_max_ratio", "0.01")
            fill("#trainingEngine", "maskrcnn_resnet50_fpn")
            wait("!!document.querySelector('#trainingAdvancedFields [data-training-param=learning_rate]')")
            click("#trainingSchedule summary")
            fill("#trainingParam-scheduler","plateau")
            assert js("document.querySelector('#trainingParam-lr_patience').getClientRects().length>0")
            assert js("document.querySelector('#trainingParam-warmup_epochs').getClientRects().length===0")
            fill("#trainingParam-scheduler","cosine")
            fill("#trainingParam-warmup_epochs","2")
            capture("31-learning-rate-settings", target="#trainingSchedule")
            click("#trainingSchedule summary")
            capture("20-training-aligned-setup")
            view.resize(1440, 1000)
            QTest.qWait(180)
            boxes = json.loads(js("JSON.stringify([...document.querySelectorAll('.training-layout > .panel')].map(e=>{const r=e.getBoundingClientRect();return [r.top,r.height,r.bottom]}))"))
            assert len(boxes) == 3, boxes
            for coordinate in (0, 1, 2):
                assert max(box[coordinate] for box in boxes) - min(box[coordinate] for box in boxes) < 2, boxes
            click(".training-config-panel .training-details summary")
            QTest.qWait(80)
            expanded = json.loads(js("JSON.stringify([...document.querySelectorAll('.training-layout > .panel')].map(e=>e.getBoundingClientRect().height))"))
            assert max(expanded) - min(expanded) < 2, expanded
            path, run, rows = saved["R004"]
            run.update(status="running", updated_at=time.time())
            atomic_json(path, run)
            click("#refreshTraining")
            wait("!document.querySelector('#backgroundTraining').hidden")
            assert js("document.querySelector('#trainingAdvancedFields [data-training-param=learning_rate]').value") == "0.003"

            # Completed models are chosen in the monitor; every recorded metric appears.
            assert not js("!!document.querySelector('#trainingMetricOptions')")
            assert set(plotted_metrics()) == {"train/loss", "val/mean_iou"}
            click("#trainingViewCompare")
            wait("document.querySelectorAll('#trainingModelPicker input[type=checkbox]').length===3")
            assert not js("!!document.querySelector('#trainingModelPicker [data-model-run=R004]')")
            for model in ("M001", "M002", "M003"):
                assert js(f"document.querySelector('#trainingModelPicker').innerText.includes('{model}')")
            set_model("R001", True)
            set_model("R002", True)
            set_model("R003", False)
            wait("document.querySelectorAll('.training-comparison-card').length===2")
            wait("new Set([...document.querySelectorAll('#trainingPlots circle[data-run-id]')].map(p=>p.dataset.runId)).size===2")
            assert js("document.querySelectorAll('#trainingPlots svg').length") == 2
            before = domains()
            assert all(float(domain[0]) == 10 for domain in before), before
            colors = json.loads(js("JSON.stringify(Object.fromEntries(['R001','R002'].map(id=>[id,[...new Set([...document.querySelectorAll('#trainingPlots circle[data-run-id]')].filter(p=>p.dataset.runId===id).map(p=>p.getAttribute('fill')))]])))"))
            assert all(len(values) == 1 for values in colors.values()), colors
            assert colors["R001"] != colors["R002"], colors
            iou = chart_script("val/mean_iou")
            # Null and absent values must not create observations at zero.
            assert js(iou + ".querySelectorAll('circle[data-run-id=R002]').length") == 5
            text = hover_epoch("val/mean_iou", 2)
            assert all(value in text for value in ("R001", "R002", "0.5800", "0.5900")), text
            missing = hover_epoch("val/mean_iou", 3)
            assert "R001" in missing and "R002" in missing and "無資料" in missing, missing
            capture("21-training-overlay", monitor=True)

            set_model("R003", True)
            wait("document.querySelectorAll('.training-comparison-card').length===3")
            wait("document.querySelector('#trainingPlots').innerText.includes('無法疊圖')")
            set_model("R003", False)
            wait("document.querySelectorAll('.training-comparison-card').length===2")
            wait("!!document.querySelector('#trainingPlots [data-metric-key=\"train/loss\"] circle[data-run-id=R001]')")
            assert domains() == before, (before, domains())
            assert not js("document.querySelector('#trainingPlots').innerText.includes('無法疊圖')")
            set_model("R001", False)
            wait("!document.querySelector('#trainingPlots circle[data-run-id=R001]')")
            assert domains() == before, (before, domains())
            assert js("document.querySelectorAll('#trainingPlots circle[data-run-id=R002]').length") > 0
            assert js("document.querySelector('#trainingPlots circle[data-run-id=R002]').getAttribute('fill')") == colors["R002"][0]
            set_model("R002", False)
            wait("!document.querySelector('#trainingPlots circle[data-run-id]')")
            QTest.qWait(1200)
            assert not js("!!document.querySelector('#trainingModelPicker input:checked')")
            set_model("R001", True)
            set_model("R002", True)
            wait("new Set([...document.querySelectorAll('#trainingPlots circle[data-run-id]')].map(p=>p.dataset.runId)).size===2")
            assert domains() == before, (before, domains())

            click('details[data-detail-key="epochs"] > summary')
            wait("document.querySelector('.training-epoch-scroll')?.clientHeight>0")
            stats = epoch_stats()
            assert stats["rows"] == 17 and stats["total"] > stats["height"], stats
            assert stats["height"] <= stats["header"] + stats["rowHeight"] * 10 + 2, stats
            assert stats["sticky"], stats
            js("document.querySelector('.training-epoch-scroll').scrollTop=120")
            QTest.qWait(1300)
            assert abs(epoch_stats()["scroll"] - 120) <= 2, epoch_stats()
            assert abs(epoch_stats()["headerTop"] - epoch_stats()["viewportTop"]) <= 2, epoch_stats()
            for key in ("evaluation", "config"):
                click(f'details[data-detail-key="{key}"] > summary')
            detail_boxes = structured("[...document.querySelectorAll('.training-detail-columns > details')].map(e=>{const r=e.getBoundingClientRect();return {top:r.top,left:r.left,width:r.width}})")
            assert len(detail_boxes) == 2 and abs(detail_boxes[0]["top"] - detail_boxes[1]["top"]) < 2, detail_boxes
            assert detail_boxes[1]["left"] >= detail_boxes[0]["left"] + detail_boxes[0]["width"], detail_boxes
            capture("24-training-epoch-details", target='details[data-detail-key="epochs"]')

            # Actual polling reads a newly appended Epoch while retaining table scroll.
            history()
            click('[data-select-run="R004"]')
            wait("document.querySelector('#trainingRunDetail').innerText.includes('訓練中')")
            wait("document.querySelector('.training-epoch-scroll tbody')?.rows.length===12")
            js("document.querySelector('.training-epoch-scroll').scrollTop=65")
            path, run, rows = saved["R004"]
            live_domains = domains()
            new_row = {"epoch": 13, "train/loss": .1714, "val/mean_iou": .9012}
            with (path.parent / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(new_row) + "\n")
            run.update(epoch=13, metrics=new_row, progress=65, updated_at=time.time())
            atomic_json(path, run)
            wait(chart_script("val/mean_iou", ".querySelectorAll('circle[data-run-id=R004]').length===13"))
            assert "0.9012" in hover_epoch("val/mean_iou", 13)
            assert domains() == live_domains, (live_domains, domains())
            assert abs(epoch_stats()["scroll"] - 65) <= 2, epoch_stats()
            run.update(status="completed", progress=100, message="訓練與評估完成", updated_at=time.time())
            atomic_json(path, run)
            click("#refreshTraining")
            wait("!document.querySelector('#trainingRunDetail').innerText.includes('訓練中')")
            history()
            click('[data-select-run="R002"]')
            wait("document.querySelector('.training-epoch-scroll tbody')?.rows.length===7")
            stats = epoch_stats()
            assert stats["total"] <= stats["height"] + 2, stats
            assert stats["height"] < stats["header"] + stats["rowHeight"] * 8, stats
            click("#trainingViewCompare")
            wait("document.querySelectorAll('.training-comparison-card').length===2")

            for width in (1440, 1024, 760):
                view.resize(width, 1000)
                QTest.qWait(180)
                capture(f"22-training-overlay-{width}", monitor=True)

            # Selecting another history item exposes only that task's available measurements.
            history()
            assert not js("!!document.querySelector('[data-compare-run]')")
            click('[data-select-run="R003"]')
            wait("!document.querySelector('#trainingHistoryDialog').open")
            wait("document.querySelectorAll('#trainingPlots svg').length===4")
            assert set(plotted_metrics()) == {"train/loss", "val/accuracy", "val/macro_f1", "val/macro_recall"}
            assert js("document.querySelector('#trainingRunDetail').innerText.includes('R003')")
            assert js(chart_script("val/macro_f1", ".querySelectorAll('circle[data-run-id=R003]').length")) == 10
            assert not js("!!document.querySelector('#trainingMetricOptions')")
            assert not page.errors, page.errors
            view.resize(1440, 1000)
            QTest.qWait(180)
            capture("23-training-classification-metrics", target="#trainingPlots")
            print("DESKTOP_TRAINING_OK", json.dumps({"project": pid, "checks": steps}, ensure_ascii=False))
        finally:
            view.close()
            service.close()


if __name__ == "__main__":
    main()
