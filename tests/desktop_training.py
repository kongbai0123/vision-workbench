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
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService
from workbench.training_engine import atomic_json


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
        assets.append({"path": str(source), "name": source.name, "split": split,
                       "batch_id": f"{split}-{index}", "review_state": "approved",
                       "source": {"kind": "test"}, "shapes": [{"id": f"part-{index}",
                       "type": "rectangle", "label": "工件" if index % 2 else "配件",
                       "x": 8, "y": 6, "width": 25, "height": 24}]})
    service.store.add_assets(pid, assets)
    snapshot = service.training.create_dataset_version(pid)
    assert snapshot["splits"] == {"train": 2, "val": 2, "test": 2}, snapshot
    saved = {}
    base = time.time() - 100
    for number in (1, 2, 3):
        run_id = f"R{number:03d}"
        classification = number == 3
        count = 7 if number == 2 else 10
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
               "config": {"epochs": 10, "device": "auto", "seed": 42 + number,
                          "image_size": 224 if classification else 640, "batch_size": 1,
                          "learning_rate": .0005}, "device": "cpu", "status": "completed",
               "message": "訓練與評估完成", "epoch": count, "progress": 100,
               "created_at": base + (0 if classification else number),
               "updated_at": base + 20, "completed_at": base + 20,
               "metrics": rows[-1], "evaluation": {
                   "validation": {"split": "val", "images": 2, **score},
                   "test": {"split": "test", "images": 2, **score}}}
        path = service.training._run_path(pid, run_id)
        path.parent.mkdir(parents=True)
        atomic_json(path, run)
        (path.parent / "metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
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
        view.resize(1440, 1000)
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

        def capture(name, monitor=False):
            if monitor:
                js("document.querySelector('#trainingRunDetail').scrollIntoView({block:'start'})")
            # Offscreen Chromium can retain a compositor frame after a dialog closes.
            # A resize invalidates the surface so the PNG represents the verified DOM.
            width, height = view.width(), view.height()
            view.resize(width + 1, height)
            QTest.qWait(80)
            view.resize(width, height)
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
            click("[data-stage=train]")
            wait("document.querySelectorAll('#trainingPlots svg').length===2")
            assert js("document.querySelector('#trainingRunDetail').innerText.includes('R002')")
            capture("20-training-aligned-setup")
            boxes = json.loads(js("JSON.stringify([...document.querySelectorAll('.training-layout > .panel')].map(e=>{const r=e.getBoundingClientRect();return [r.top,r.height,r.bottom]}))"))
            assert len(boxes) == 3, boxes
            for coordinate in (0, 1, 2):
                assert max(box[coordinate] for box in boxes) - min(box[coordinate] for box in boxes) < 2, boxes
            click(".training-config-panel .training-details summary")
            QTest.qWait(80)
            expanded = json.loads(js("JSON.stringify([...document.querySelectorAll('.training-layout > .panel')].map(e=>e.getBoundingClientRect().height))"))
            assert max(expanded) - min(expanded) < 2, expanded

            # Available metrics are collected values, not every metric in the model catalog.
            options = js("document.querySelector('#trainingMetricOptions').innerText")
            assert "IoU" in options and "Loss" in options, options
            assert "mAP" not in options and "Accuracy" not in options, options
            history()
            click('[data-compare-run="R001"]')
            click('[data-compare-run="R002"]')
            if js("document.querySelector('#trainingHistoryDialog').open"):
                click("#closeTrainingHistory")
            click("#trainingViewCompare")
            wait("document.querySelectorAll('#trainingRunDetail [data-toggle-run]').length===2")
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

            click('[data-toggle-run="R001"]')
            wait("!document.querySelector('#trainingPlots circle[data-run-id=R001]')")
            assert domains() == before, (before, domains())
            assert js("document.querySelectorAll('#trainingPlots circle[data-run-id=R002]').length") > 0
            assert js("document.querySelector('#trainingPlots circle[data-run-id=R002]').getAttribute('fill')") == colors["R002"][0]
            click('[data-toggle-run="R002"]')
            wait("!document.querySelector('#trainingPlots circle[data-run-id]')")
            assert domains() == before, (before, domains())
            click('[data-toggle-run="R001"]')
            click('[data-toggle-run="R002"]')
            wait("new Set([...document.querySelectorAll('#trainingPlots circle[data-run-id]')].map(p=>p.dataset.runId)).size===2")

            # Actual polling reads a newly appended Epoch while retaining comparison state.
            path, run, rows = saved["R002"]
            run.update(status="running", progress=70, message="Epoch 7 / 10", updated_at=time.time())
            atomic_json(path, run)
            click("#refreshTraining")
            wait("document.querySelector('#trainingRunDetail').innerText.includes('訓練中')")
            new_row = {"epoch": 8, "train/loss": .1714, "val/mean_iou": .9012}
            with (path.parent / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(new_row) + "\n")
            run.update(epoch=8, metrics=new_row, progress=80, updated_at=time.time())
            atomic_json(path, run)
            wait(chart_script("val/mean_iou", ".querySelectorAll('circle[data-run-id=R002]').length===6"))
            assert "0.9012" in hover_epoch("val/mean_iou", 8)
            assert domains() == before, (before, domains())
            wait("[...document.querySelectorAll('.training-comparison-card')].some(e=>e.innerText.includes('R002')&&e.innerText.includes('Epoch 8 / 10'))")
            run.update(status="completed", progress=100, message="訓練與評估完成", updated_at=time.time())
            atomic_json(path, run)
            click("#refreshTraining")
            wait("!document.querySelector('#trainingRunDetail').innerText.includes('訓練中')")

            for width in (1440, 1024, 760):
                view.resize(width, 1000)
                QTest.qWait(180)
                capture(f"22-training-overlay-{width}", monitor=True)

            # Selecting another history item exposes only that task's available measurements.
            history()
            click('[data-select-run="R003"]')
            wait("!document.querySelector('#trainingHistoryDialog').open")
            wait("document.querySelector('#trainingMetricOptions').innerText.includes('Accuracy')")
            options = js("document.querySelector('#trainingMetricOptions').innerText")
            assert "F1" in options and "Recall" in options and "IoU" not in options, options
            assert js("document.querySelector('#trainingRunDetail').innerText.includes('R003')")
            click('#trainingMetricOptions input[data-metric-key="val/macro_f1"]')
            wait("document.querySelectorAll('#trainingPlots svg').length===3")
            assert js(chart_script("val/macro_f1", ".querySelectorAll('circle[data-run-id=R003]').length")) == 10
            click('#trainingMetricOptions input[data-metric-key="train/loss"]')
            wait("document.querySelectorAll('#trainingPlots svg').length===2")
            assert not js("!!document.querySelector('#trainingPlots [data-metric-key=\"train/loss\"]')")
            assert not page.errors, page.errors
            view.resize(1440, 1000)
            capture("23-training-classification-metrics", monitor=True)
            print("DESKTOP_TRAINING_OK", json.dumps({"project": pid, "checks": steps}, ensure_ascii=False))
        finally:
            view.close()
            service.close()


if __name__ == "__main__":
    main()
