"""Exercise the project material toolbar with a real catalog and a Qt viewport."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.materials import record_annotation, record_video_frame
from workbench.server import WorkbenchService


def seed_project(service, root):
    project_id = service.store.create_project("素材追溯驗收")["id"]
    directory = service.store.directory(project_id)
    image = root / "握把第 5 幀.png"
    Image.new("RGB", (64, 48), (36, 90, 148)).save(image)
    image_id, = service.store.add_assets(project_id, [{"path": str(image)}])["asset_ids"]

    video = directory / "inputs/videos" / "recording_test.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"recording fixture for material catalog")
    entry = service.materials.register_video(project_id, {
        "path": str(video), "display_name": "握把產線.mp4", "fps": 10, "frames": 10,
    })
    data = b'{"shapes":[]}'
    checksum = sha256(data).hexdigest()
    annotation = directory / "inputs/annotations" / f"{checksum}.json"
    annotation.parent.mkdir()
    annotation.write_bytes(data)
    with service.store.connection(project_id, write=True) as db:
        annotation_id = record_annotation(
            db, project_id, relative_path=annotation.relative_to(directory).as_posix(),
            sha256=checksum, byte_count=len(data), name="握把標註.json", format="labelme",
            image_id=image_id, annotation_revision=1,
        )
        record_video_frame(db, image_id, {
            "video_id": entry["id"], "video_sha256": entry["sha256"], "frame_index": 5,
            "timestamp_seconds": .5, "timestamp_method": "container",
        })
    return project_id, directory, {"image": image_id, "video": entry["id"], "annotation": annotation_id}


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "desktop"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision-project-materials-") as folder:
        root = Path(folder)
        service = WorkbenchService(root / "data")
        service.start()
        view = QWebEngineView()
        view.resize(1440, 900)
        view.show()
        page = view.page()

        def js(script):
            result = []
            loop = QEventLoop()
            page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(8000, loop.quit)
            loop.exec()
            assert result, script
            return result[0]

        def wait(expression, timeout=12):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if js(expression):
                    return
                QTest.qWait(50)
            raise AssertionError(f"Timed out: {expression}\n{js('document.body.innerText.slice(0, 1500)')}")

        def click(selector):
            node = f"document.querySelector({json.dumps(selector)})"
            assert js(f"!!{node} && !{node}.disabled && {node}.getClientRects().length > 0"), selector
            js(f"{node}.click()")

        def capture(name, probe_selector=None):
            js("""document.body.dataset.qaPaint='pending';
                requestAnimationFrame(()=>requestAnimationFrame(()=>{document.body.dataset.qaPaint='ready'}));
            """)
            wait("document.body.dataset.qaPaint==='ready'")
            QTest.qWait(280)
            view.repaint()
            snapshot = view.grab()
            if probe_selector:
                bounds = json.loads(js(f"""JSON.stringify(document
                    .querySelector({json.dumps(probe_selector)}).getBoundingClientRect().toJSON())"""))
                deadline = time.monotonic() + 4
                while True:
                    image = snapshot.toImage()
                    ratio = image.width() / view.width()
                    if any(image.pixelColor(x, y).lightness() > 120
                           for y in range(round(bounds["top"] * ratio), round(bounds["bottom"] * ratio))
                           for x in range(round(bounds["left"] * ratio), round(bounds["right"] * ratio))):
                        break
                    assert time.monotonic() < deadline, f"Material name not painted in {name}"
                    QTest.qWait(120)
                    snapshot = view.grab()
            assert snapshot.save(str(output / f"{name}.png")), name

        def row(kind):
            selector = f'#projectMaterialList [data-material-id="{material_ids[kind]}"]'
            return f"document.querySelector({json.dumps(selector)})"

        try:
            project_id, directory, material_ids = seed_project(service, root)
            view.setUrl(QUrl(service.entry_url))
            wait("document.getElementById('workspaceLoading')?.dataset.startup === 'settled' && !!document.querySelector('.project-card-open')")
            for selector, label in [
                ("#projectMaterials", "素材清單"),
                ("#dataExport", "匯出資料集"),
                ("#openProjectFolder", "專案資料夾"),
            ]:
                control = f"document.querySelector({json.dumps(selector)})"
                assert js(f"{control}.textContent.includes({json.dumps(label)}) && {control}.disabled"), selector
            click(".project-card-open")
            wait(f"window.workbenchState().projectId === {json.dumps(project_id)} && !window.workbenchState().transitioning")
            capture("36a-project-materials-toolbar")

            click("#projectMaterials")
            wait("document.querySelector('#projectMaterialsDialog').open && "
                 "document.querySelectorAll('#projectMaterialList .project-material-row').length === 3")
            image_text = js(f"{row('image')}.innerText")
            assert "握把第 5 幀.png" in image_text, image_text
            assert all(part in image_text for part in ("握把產線.mp4", "第 5 幀", "0.5 秒")), image_text
            annotation_text = js(f"{row('annotation')}.innerText")
            assert all(part in annotation_text for part in ("握把標註.json", "握把第 5 幀.png", "修訂 1")), annotation_text
            capture("36-project-materials", "#projectMaterialList .project-material-row b")
            for kind in material_ids:
                click(f'#projectMaterialFilters [data-material-kind="{kind}"]')
                wait(f"!!{row(kind)} && document.querySelectorAll('#projectMaterialList .project-material-row').length === 1")
            click('#projectMaterialFilters [data-material-kind="all"]')
            js("""const search = document.querySelector('#projectMaterialSearch');
                search.value = '握把產線.mp4';
                search.dispatchEvent(new Event('input', {bubbles: true}));
            """)
            wait(f"!!{row('video')} && !{row('annotation')}")
            click("#closeProjectMaterials")
            wait("!document.querySelector('#projectMaterialsDialog').open")

            click("#dataExport")
            wait("!document.querySelector('#export').hidden && !window.workbenchState().transitioning && "
                 "document.querySelector('#exportFormat').value === 'yolo_detection'")
            click("#backFromExport")
            wait("!document.querySelector('#annotate').hidden && !window.workbenchState().transitioning")
            if os.name == "nt":
                with patch("workbench.server.os.startfile") as open_native:
                    click("#openProjectFolder")
                    deadline = time.monotonic() + 5
                    while not open_native.called and time.monotonic() < deadline:
                        QTest.qWait(50)
                    open_native.assert_called_once_with(directory)
            print("DESKTOP_PROJECT_MATERIALS_OK", flush=True)
        finally:
            view.close()
            page.deleteLater()
            QTest.qWait(50)
            service.close()


if __name__ == "__main__":
    main()
