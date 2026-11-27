"""Exercise the project library using the desktop Chromium viewport and a temporary store."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlsplit

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer, Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService


def seed_projects(store, source):
    def add(name, states, classes=()):
        project = store.create_project(name)
        if classes:
            store.update_project(project["id"], classes=list(classes))
        records = []
        for index, state in enumerate(states):
            path = source / f"{index}.png"
            Image.new("RGB", (32, 24), (40 + index, 70, 120)).save(path)
            records.append({"path": path, "review_state": state})
        if records:
            store.add_assets(project["id"], records)
        return project["id"]

    return {
        "empty": add("新批次樣本", []),
        "excluded": add("排除樣本", ["rejected"] * 3, ["瑕疵"]),
        "reviewed": add("握把覆核", ["approved"] * 8 + ["rejected"] * 2,
                        ["grip", "grip_done", "mid", "mid_done", "handlebar", "hand"]),
        "pending": add("零件外觀檢查", ["approved"] * 2 + ["pending"] * 6, ["工件", "刮痕"]),
    }


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "desktop"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision-project-library-") as folder:
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
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            if not result:
                raise AssertionError("JavaScript timed out")
            return result[0]

        def wait(expression, seconds=20):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(expression):
                    return
                QTest.qWait(50)
            raise AssertionError(f"Timed out: {expression}\n{js('document.body.innerText.slice(0, 2000)')}")

        def click(selector):
            node = f"document.querySelector({json.dumps(selector)})"
            assert js(f"!!{node} && !{node}.disabled && {node}.getClientRects().length > 0"), selector
            js(f"{node}.click()")

        def input_text(selector, text):
            js(f"""{{
                const input=document.querySelector({json.dumps(selector)});
                input.value={json.dumps(text)};
                input.dispatchEvent(new Event('input', {{bubbles:true}}));
            }}""")

        def capture(name, dialog_visible=False):
            js("""document.body.dataset.qaPaint='pending';
                requestAnimationFrame(()=>requestAnimationFrame(()=>{document.body.dataset.qaPaint='ready'}));
            """)
            wait("document.body.dataset.qaPaint==='ready'")
            QTest.qWait(320)
            view.repaint()
            snapshot = view.grab()
            if dialog_visible:
                # Chromium can still return the last composited frame on the first grab.
                probe = json.loads(js(r"""JSON.stringify((()=>{
                    const dialog=document.querySelector('#formDialog');
                    const rect=dialog.getBoundingClientRect();
                    const rgb=getComputedStyle(dialog).backgroundColor.match(/\d+/g).slice(0,3).map(Number);
                    return [rect.left+rect.width/2, rect.top+12, ...rgb];
                })())"""))
                deadline = time.monotonic() + 4
                while True:
                    width_ratio = snapshot.width() / view.width()
                    color = snapshot.toImage().pixelColor(round(probe[0]*width_ratio), round(probe[1]*width_ratio))
                    if max(abs(channel-target) for channel, target in zip(color.getRgb(), probe[2:])) <= 6:
                        break
                    assert time.monotonic() < deadline, f"Dialog not painted in {name}"
                    QTest.qWait(150)
                    snapshot = view.grab()
            assert snapshot.save(str(output / f"{name}.png")), name

        def card(name):
            return f"""[...document.querySelectorAll('#projectGrid .project-card')]
                .find(item=>item.querySelector('h2').textContent==={json.dumps(name)})"""

        def card_text(name):
            return js(f"({card(name)}).innerText")

        def verify_review(name, state, reviewed=None, total=None):
            assert js(f"({card(name)}).querySelector('.project-status').dataset.reviewStatus") == state
            progress = f"({card(name)}).querySelector('progress.project-review-progress')"
            if total is None:
                assert js(f"{progress}===null")
            else:
                assert js(f"{progress}.value==={reviewed} && {progress}.max==={total}")

        def verify_layout(summary_columns):
            layout = json.loads(js("""JSON.stringify((()=>{
                const bounds=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
                const library=document.querySelector('#library');
                const workflow=document.querySelector('.workflow-navigation .phasebar');
                return {
                    grid:bounds('#projectGrid'), summary:bounds('.library-summary'),
                    toolbar:bounds('#library .section-heading'),
                    overflow:library.scrollWidth-library.clientWidth,
                    workflowVerticalScrollbar:workflow.offsetWidth-workflow.clientWidth,
                    metrics:[...library.querySelectorAll('.library-summary > div')].map(
                        metric=>metric.getBoundingClientRect().toJSON()),
                    cards:[...document.querySelectorAll('#projectGrid .project-card')].map(
                        card=>card.getBoundingClientRect().toJSON()),
                };
            })())"""))
            for area in ("summary", "toolbar"):
                for edge in ("left", "right"):
                    assert abs(layout[area][edge] - layout["grid"][edge]) <= 1, layout
            assert layout["overflow"] <= 2, layout
            assert layout["workflowVerticalScrollbar"] <= 1, layout
            assert all(card["left"] >= layout["grid"]["left"] - 1 and
                       card["right"] <= layout["grid"]["right"] + 1 for card in layout["cards"]), layout
            metrics = layout["metrics"]
            assert len(metrics) == 4, layout
            assert all(abs(metrics[0]["width"] - metric["width"]) <= 1 for metric in metrics), layout
            assert sum(abs(metrics[0]["top"] - metric["top"]) <= 1 for metric in metrics) == summary_columns, layout

        try:
            projects = seed_projects(service.store, root)
            view.setUrl(QUrl(service.entry_url))
            wait("document.querySelectorAll('#projectGrid .project-card').length===4")
            wait("document.querySelector('#workspaceLoading').hidden")

            assert "尚未匯入" in card_text("新批次樣本")
            assert "待審核" in card_text("零件外觀檢查")
            assert "審核完成" in card_text("握把覆核")
            assert "已排除" in card_text("排除樣本")
            assert "可訓練" not in card_text("握把覆核") + card_text("排除樣本")
            verify_review("新批次樣本", "empty")
            verify_review("零件外觀檢查", "pending", 2, 8)
            verify_review("握把覆核", "complete", 10, 10)
            verify_review("排除樣本", "excluded", 3, 3)
            assert js(f"({card('握把覆核')}).querySelector('.project-class-more').textContent.includes('+4')")
            assert js("document.querySelector('#projectCount').textContent.trim()") == "4"
            assert js("document.querySelector('#totalImages').textContent.trim()") == "21"
            assert js("document.querySelector('#totalApproved').textContent.trim()") == "10"
            assert js("document.querySelector('#totalPending').textContent.trim()") == "6"
            assert js("document.querySelector('[data-stage=train]').disabled")
            assert js("!document.querySelector('#libraryWorkflowHint').hidden")

            verify_layout(4)
            capture("01a-project-library-overview")
            view.resize(1024, 760)
            QTest.qWait(120)
            verify_layout(4)
            view.resize(480, 760)
            QTest.qWait(120)
            verify_layout(2)
            assert js("""(()=>{
                const nav=document.querySelector('.workflow-navigation .phasebar');
                nav.scrollLeft=100;
                const scrollable=nav.scrollLeft>0;
                nav.scrollLeft=0;
                return scrollable;
            })()""")
            capture("01b-project-library-narrow")
            view.resize(1440, 900)
            QTest.qWait(120)

            input_text("#projectSearch", "握把")
            assert js("document.querySelectorAll('#projectGrid .project-card').length") == 1
            assert "握把覆核" in card_text("握把覆核")
            assert js("document.querySelector('#projectCount').textContent.trim()") == "4"
            assert js("document.querySelector('#totalPending').textContent.trim()") == "6"
            input_text("#projectSearch", "無相符項目")
            assert js("!document.querySelector('#libraryEmpty').hidden && !document.querySelector('#projectGrid').children.length")
            input_text("#projectSearch", "")

            delete = f"({card('零件外觀檢查')}).querySelector('button.project-delete')"
            assert js(f"{delete}.getAttribute('aria-label')") == "刪除專案「零件外觀檢查」"
            assert js(f"{delete}.focus();document.activeElement==={delete}")
            QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Return)
            wait("document.querySelector('#formDialog').open")
            assert "零件外觀檢查" in js("document.querySelector('#dialogBody').innerText")
            assert "8 張影像" in js("document.querySelector('#dialogBody').innerText")
            capture("01c-project-delete-confirmation", dialog_visible=True)
            click("#cancelDialog")
            wait("!document.querySelector('#formDialog').open")
            assert len(service.store.list_projects()) == 4
            assert js("!document.querySelector('#library').hidden")

            js(f"({card('握把覆核')}).querySelector('button.project-card-open').click()")
            wait("!document.querySelector('#projectName').disabled")
            assert js("document.querySelector('#libraryWorkflowHint').hidden")
            assert js("document.querySelector('#projectName').innerText.includes('握把覆核')")
            assert js("""(()=>{
                const name=document.querySelector('#projectName');
                const hint=document.getElementById(name.getAttribute('aria-describedby'));
                return name.classList.contains('can-rename') && hint?.textContent.includes('編輯專案名稱') &&
                       getComputedStyle(name,'::after').content.includes('✎');
            })()""")
            assert js("!document.querySelector('[data-stage=train]').disabled")
            click("#projectName")
            wait("document.querySelector('#formDialog').open")
            assert js("document.querySelector('#dialogTitle').innerText") == "編輯專案名稱"
            input_text("#dialogBody input", "握把覆核已改名")
            click("#confirmDialog")
            wait("!document.querySelector('#formDialog').open")
            assert service.store.get_project(projects["reviewed"])["name"] == "握把覆核已改名"

            def open_creation_dialog(name):
                click("#home")
                wait("!document.querySelector('#library').hidden && !document.querySelector('#home').disabled")
                click("#newProject")
                wait("document.querySelector('#formDialog').open")
                input_text("#createProjectName", name)
                assert js("!document.querySelector('#createProjectClasses').required")

            def submit_creation(name, classes):
                click("#confirmDialog")
                expected_classes = json.dumps(classes)
                wait(f"""!document.querySelector('#projectName').disabled &&
                         document.querySelector('#projectName').textContent==={json.dumps(name)} &&
                         JSON.stringify([...document.querySelectorAll('#annotationClassChoices button')].map(
                             node=>node.dataset.className))===JSON.stringify({expected_classes})""")
                created = next(project for project in service.store.list_projects() if project["name"] == name)
                assert created["classes"] == classes, created

            open_creation_dialog("握把新批次")
            input_text("#createProjectClasses", "grip\nmid\ngrip_done\nmid_done\nhand")
            capture("01d-project-library-create-classes", dialog_visible=True)
            input_text("#createProjectClasses", "  grip\nmid; grip_done；mid_done\nhand；grip  ")
            submit_creation("握把新批次", ["grip", "mid", "grip_done", "mid_done", "hand"])
            open_creation_dialog("尚未指定類別")
            assert js("document.querySelector('#createProjectClasses').value===''")
            assert js(f"({card('握把新批次')}).querySelector('.project-classes').title.includes('grip_done')")
            assert "+3" in card_text("握把新批次")
            submit_creation("尚未指定類別", [])

            click("#home")
            wait("!document.querySelector('#library').hidden && !document.querySelector('#home').disabled")
            backup = f"({card('握把覆核已改名')}).querySelector('button.project-card-backup')"
            assert js(f"{backup}.getAttribute('aria-label')") == "備份專案「握把覆核已改名」"
            js(f"{backup}.click()")
            wait("document.querySelector('#formDialog').open")
            assert js("document.querySelector('#dialogTitle').innerText") == "備份專案"
            capture("01e-project-backup", dialog_visible=True)
            click("#confirmDialog")
            wait("document.querySelector('#projectTransferResult code')?.textContent.endsWith('.vwb-project.zip')")
            archive = Path(js("document.querySelector('#projectTransferResult code').textContent"))
            assert archive.is_file(), archive

            target = WorkbenchService(root / "restored-data")
            target.start()
            try:
                target_url = target.entry_url
                target_port = json.dumps(str(urlsplit(target_url).port))
                view.setUrl(QUrl(target_url))
                wait(f"location.port==={target_port} && document.getElementById('workspaceLoading')?.dataset.startup==='settled' && document.querySelector('#projectCount')?.textContent==='0'")
                click("#restoreProjectBackup")
                wait("document.querySelector('#formDialog').open")
                click("#cancelDialog")
                wait("!document.querySelector('#formDialog').open && !document.querySelector('#restoreProjectBackup').disabled")
                assert target.store.list_projects() == []
                click("#restoreProjectBackup")
                wait("document.querySelector('#formDialog').open")
                input_text("#projectTransferPath", str(archive))
                click("#confirmDialog")
                wait("document.querySelector('#projectTransferResult strong')?.textContent.includes('還原完成') && document.querySelector('#projectGrid h2')?.textContent==='握把覆核已改名'")
                restored = target.store.get_project(projects["reviewed"])
                assert restored["classes"] == ["grip", "grip_done", "mid", "mid_done", "handlebar", "hand"]
                assert len(restored["assets"]) == 10
                assert js("!document.querySelector('#library').hidden && document.querySelector('#projectName').disabled")
                capture("01f-project-restore-result")
                click("#projectTransferResult button")
                wait("document.querySelector('#projectName').textContent==='握把覆核已改名' && !document.querySelector('#projectName').disabled")
            finally:
                target.close()
            print("DESKTOP_PROJECT_LIBRARY_OK: 4 initial projects, 3 viewport sizes, search, delete cancel, rename, optional classes and cross-root backup/restore")
        finally:
            view.close()
            service.close()


if __name__ == "__main__":
    main()
