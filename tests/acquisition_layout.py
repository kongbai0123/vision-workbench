"""Explicit Qt Chromium acquisition-layout acceptance with synthetic local data.

Run: .venv/Scripts/python.exe tests/acquisition_layout.py
Scope: isolated temporary project, 52 synthetic images, UI layout and import.
Does not start a camera, capture the screen, or access the user's project data.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding",
)

from pathlib import Path
import json
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService


class Page(QWebEnginePage):
    def __init__(self, *args):
        super().__init__(*args)
        self.errors = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.errors.append(f"{source}:{line}: {message}")
            print("WEB:", message, flush=True)


SOURCES = ("camera", "files", "video", "screen", "merge")
SIZES = ((1280, 720), (1440, 900), (1910, 992), (1920, 1080))
IMAGE_COUNT = 52
THUMBNAIL_COUNT = IMAGE_COUNT


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "acquisition-layout"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "success": False,
        "scope": "Synthetic data only; temporary isolated project; no physical camera or screen capture",
        "source_tabs": list(SOURCES),
        "project_images": 0,
        "steps": [],
        "layout": [],
        "page_errors": [],
        "violations": [],
    }

    with tempfile.TemporaryDirectory(prefix="vision-acquisition-layout-") as directory:
        root = Path(directory)
        source = root / "合成驗收影像"
        source.mkdir()
        for index in range(IMAGE_COUNT):
            image = Image.new("RGB", (960, 640), (217, 227, 230))
            draw = ImageDraw.Draw(image)
            offset = (index % 3) * 40
            draw.rounded_rectangle(
                (190 + offset, 100, 700, 550 - offset),
                radius=30,
                fill=(40 + (index % 3) * 25, 90 + (index % 3) * 20, 130),
            )
            draw.ellipse((350, 245, 530, 425), fill=(217, 227, 230))
            draw.text((30, 30), f"SYNTHETIC FIXTURE {index + 1}", fill=(30, 45, 55))
            image.save(source / f"collection_session_20260909_工業檢測合成工件_長檔名驗收_{index + 1:03d}.png")

        service = WorkbenchService(root / "data").start()
        view = QWebEngineView()
        page = Page(view)
        view.setPage(page)
        view.resize(1440, 900)
        view.show()
        view.setUrl(QUrl(service.entry_url))
        page.setVisible(True)

        def js(script):
            result = []
            loop = QEventLoop()
            page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            if not result:
                raise AssertionError("JavaScript timed out")
            return result[0]

        def wait(expression, seconds=40):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if js(expression):
                    return
                QTest.qWait(80)
            raise AssertionError(
                "Timeout: " + expression + "\n" + str(js("document.body.innerText.slice(-4000)"))
            )

        def click(selector):
            encoded = json.dumps(selector)
            wait("typeof window.workbenchState==='function' && !window.workbenchState().busy && !window.workbenchState().transitioning")
            wait(f"!!document.querySelector({encoded}) && !document.querySelector({encoded}).disabled")
            js(f"document.querySelector({encoded}).click()")

        def fill(selector, value):
            js(
                "{const e=document.querySelector("
                + json.dumps(selector)
                + ");e.value="
                + json.dumps(value)
                + ";e.dispatchEvent(new Event('input',{bubbles:true}));"
                "e.dispatchEvent(new Event('change',{bubbles:true}));}"
            )

        def select_source(name):
            click(f"[data-source='{name}']")
            wait(
                "(()=>{const e=document.querySelector('#source-"
                + name
                + "');return e && e.getAttribute('role')==='tabpanel' && e.getClientRects().length>0})()"
            )
            shown = json.loads(
                js(
                    "JSON.stringify("
                    + json.dumps(list(SOURCES))
                    + ".filter(name=>document.querySelector('#source-'+name)?.getClientRects().length>0))"
                )
            )
            assert shown == [name], f"Expected one source panel {name}, got {shown}"

        def capture(name, source_name, expected_thumbnails):
            QTest.qWait(220)
            view.grab().save(str(output / f"{name}.png"))
            result = json.loads(
                js(
                    """JSON.stringify((()=>{
                      const visible=e=>e&&e.getClientRects().length>0;
                      const panels=[document.documentElement,document.body,document.querySelector('#acquire'),
                        ...document.querySelectorAll('.source-workspace,.acquired-tray,[role=tabpanel],.source-inspector')].filter(visible);
                      const overflows=panels.filter(e=>e.scrollWidth>e.clientWidth+2)
                        .map(e=>({element:e.id||e.className||e.tagName,width:e.clientWidth,scroll:e.scrollWidth}));
                      const bounds=e=>e?.getBoundingClientRect().toJSON();
                      const preview=document.querySelector('#source-camera .camera-preview');
                      const viewportVisible=e=>{if(!visible(e))return false;const r=e.getBoundingClientRect();
                        if(r.width<=0||r.height<=0||r.left<0||r.top<0||r.right>innerWidth+1||r.bottom>innerHeight+1)return false;
                        for(let p=e.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),q=p.getBoundingClientRect();
                          if(/auto|scroll|hidden|clip/.test(s.overflowY)&&(r.top<q.top-1||r.bottom>q.bottom+1))return false;
                          if(/auto|scroll|hidden|clip/.test(s.overflowX)&&(r.left<q.left-1||r.right>q.right+1))return false;}
                        return true;};
                      const active=document.querySelector('[role=tabpanel]:not([hidden])');
                      const workspace=document.querySelector('.source-workspace');
                      const tray=document.querySelector('.acquired-tray');
                      const list=document.querySelector('#acquiredAssets');
                      const compactCapture=document.querySelector('.compact-camera-capture');
                      const acquiredHeading=document.querySelector('.acquired-heading');
                      const selectAllLabel=document.querySelector('#selectAllAssetsLabel');
                      const annotateButton=document.querySelector('#goAnnotate');
                      const autoInterval=document.querySelector('#autoCaptureInterval');
                      const autoLimit=document.querySelector('#autoCaptureLimit');
                      const startAuto=document.querySelector('#startAutoCapture');
                      const startRecording=document.querySelector('#startRecording');
                      const inspector=document.querySelector('#source-camera .source-inspector');
                      const first=list.querySelector('.acquired-item'),last=list.querySelector('.acquired-item:last-child');
                      list.scrollTop=0;
                      const initialLast=bounds(last);
                      list.scrollTop=list.scrollHeight;
                      const finalLast=bounds(last),r=finalLast;
                      const hit=r?document.elementFromPoint(r.left+r.width/2,r.top+r.height/2):null;
                      const lastHit=!!last&&!!hit&&(hit===last||last.contains(hit));
                      const scroller={bounds:bounds(list),count:list.querySelectorAll('.acquired-item').length,
                        width:list.clientWidth,height:list.clientHeight,scroll_height:list.scrollHeight,overflowY:getComputedStyle(list).overflowY,
                        end_scroll_top:list.scrollTop,last_initial:initialLast,last_final:finalLast,
                        last_visible:viewportVisible(last),last_hit:lastHit,
                        parent_height:tray.clientHeight,parent_scroll_height:tray.scrollHeight};
                      list.scrollTop=0;
                      return {width:innerWidth,height:innerHeight,overflows,
                        workspace:bounds(workspace),active_panel:bounds(active),tray:bounds(tray),
                        workspace_visible:viewportVisible(workspace),active_panel_visible:viewportVisible(active),
                        tray_visible:viewportVisible(tray),scroller,
                        camera_inspector:visible(inspector)?bounds(inspector):null,
                        camera_inspector_visible:viewportVisible(inspector),
                        inspector_open:workspace.classList.contains('inspector-open'),
                        inspector_toggle_visible:viewportVisible(document.querySelector('#toggleSourceInspector')),
                        compact_capture:visible(compactCapture)?bounds(compactCapture):null,
                        compact_capture_visible:viewportVisible(compactCapture),
                        acquired_heading:bounds(acquiredHeading),
                        select_all:visible(selectAllLabel)?bounds(selectAllLabel):null,
                        annotate_button:visible(annotateButton)?bounds(annotateButton):null,
                        auto_interval:visible(autoInterval)?bounds(autoInterval):null,
                        auto_limit:visible(autoLimit)?bounds(autoLimit):null,
                        start_auto:visible(startAuto)?bounds(startAuto):null,
                        start_recording:visible(startRecording)?bounds(startRecording):null,
                        preview:visible(preview)?bounds(preview):null,
                        start_button_visible:viewportVisible(document.querySelector('#startCamera')),
                        capture_button_visible:viewportVisible(document.querySelector('#takeSnapshot')),
                        annotate_button_visible:viewportVisible(document.querySelector('#goAnnotate')),
                        import_button_visible:viewportVisible(document.querySelector('#importFiles'))};
                    })())"""
                )
            )
            result.update({"source": source_name, "screenshot": f"{name}.png"})
            report["layout"].append(result)
            issues = []
            if result["overflows"]:
                issues.append(f"Horizontal overflow: {result['overflows']}")
            for target in ("workspace", "active_panel", "tray"):
                if not result[f"{target}_visible"]:
                    issues.append(f"{target} outside viewport: {result[target]}")
            ratio=result["workspace"]["width"]/result["tray"]["width"]
            if not 2.85 <= ratio <= 3.15:
                issues.append(f"Output/assets width ratio must remain near 3:1, got {ratio:.3f}")
            if not result["annotate_button_visible"]:
                issues.append("Annotation button clipped")
            if result["select_all"] and result["annotate_button"]:
                select_mid=result["select_all"]["top"]+result["select_all"]["height"]/2
                annotate_mid=result["annotate_button"]["top"]+result["annotate_button"]["height"]/2
                if abs(select_mid-annotate_mid) > 3:
                    issues.append("Select-all and annotation controls must share one horizontal row")
            if result["auto_interval"] and result["auto_limit"]:
                if abs(result["auto_interval"]["top"]-result["auto_limit"]["top"]) > 1:
                    issues.append("Auto-capture inputs are vertically misaligned")
                if result["start_auto"] and result["start_recording"] and abs(result["start_auto"]["top"]-result["start_recording"]["top"]) > 1:
                    issues.append("Auto-capture and recording buttons are vertically misaligned")
            scroller = result["scroller"]
            if scroller["count"] != expected_thumbnails:
                issues.append(f"Expected {expected_thumbnails} thumbnails, got {scroller['count']}")
            if expected_thumbnails:
                if scroller["overflowY"] not in ("auto", "scroll") or scroller["end_scroll_top"] <= 0:
                    issues.append(f"Thumbnail column must scroll internally: {scroller}")
                if not scroller["last_visible"] or not scroller["last_hit"]:
                    issues.append("Last thumbnail cannot be fully exposed and hit after vertical scroll")
                if scroller["parent_scroll_height"] > scroller["parent_height"] + 2:
                    issues.append("Thumbnail scroller expanded its parent")
            if source_name == "camera":
                if not result["preview"] or result["preview"]["width"] < 600:
                    issues.append("Camera preview too narrow")
                if result["camera_inspector_visible"] or result["inspector_open"]:
                    issues.append("Camera settings drawer should be hidden by default")
                if not result["inspector_toggle_visible"]:
                    issues.append("Settings drawer control clipped")
                if not result["compact_capture_visible"]:
                    issues.append("Compact auto-capture/recording card is not visible for camera")
                elif result["compact_capture"]["bottom"] > result["acquired_heading"]["top"] + 1:
                    issues.append("Compact capture card must sit above the project assets heading")
                elif result["compact_capture"]["height"] > 180:
                    issues.append(f"Compact capture card is too tall: {result['compact_capture']['height']}")
            elif result["compact_capture_visible"]:
                issues.append("Camera capture card leaked into another acquisition source")
            report["steps"].append(name)
            report["violations"].extend({"case": name, "message": issue} for issue in issues)
            print("FAIL" if issues else "PASS", name, flush=True)
            if issues:
                print(json.dumps({"viewport": [result["width"], result["height"]],
                    "workspace_right": result["workspace"]["right"],
                    "active_panel_right": result["active_panel"]["right"],
                    "tray_height": scroller["height"], "tray_scroll_height": scroller["scroll_height"],
                    "issues": issues}, ensure_ascii=False), flush=True)

        try:
            wait("typeof window.workbenchState==='function'")
            wait("!document.querySelector('#libraryEmpty').hidden")
            click("#newProject")
            wait("document.querySelector('#formDialog').open")
            fill("#dialogBody input", "採集版面驗收 · 合成資料")
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("!document.querySelector('#acquire').hidden && !window.workbenchState().transitioning")
            duplicate_ids = json.loads(js("JSON.stringify((()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return [...new Set(ids.filter((id,index)=>ids.indexOf(id)!==index))]})())"))
            assert not duplicate_ids, f"Duplicate DOM ids: {duplicate_ids}"
            assert js("document.querySelector('#source-camera').getClientRects().length>0"), "Camera is not the default source"
            assert not js("document.querySelector('#source-camera .source-inspector').getClientRects().length"), "Settings drawer must default to hidden"
            assert js("!!document.querySelector('#showCenterCrosshair')"), "Center crosshair toggle is missing"
            js("document.querySelector('#showCenterCrosshair').click()")
            assert js("document.querySelector('#cameraPreview').classList.contains('show-crosshair')"), "Center crosshair did not turn on"
            js("document.querySelector('#showCenterCrosshair').click()")
            assert not js("document.querySelector('#cameraPreview').classList.contains('show-crosshair')"), "Center crosshair did not turn off"
            report["steps"].append("center-crosshair-toggle")
            js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'q',bubbles:true}))")
            assert js("document.querySelector('#cameraTargetTool').value") == "rectangle", "Q did not select Bounding Box"
            assert js("document.querySelector('[data-target-hint=rectangle]').classList.contains('active')"), "Selected shortcut is not visually indicated"
            js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'w',bubbles:true}))")
            assert js("document.querySelector('#cameraTargetTool').value") == "polygon", "W did not select Polygon"
            js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'E',shiftKey:true,bubbles:true}))")
            assert js("document.querySelector('#cameraTargetTool').value") == "mask_erase", "Shift+E did not select eraser"
            js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))")
            assert js("document.querySelector('#cameraTargetTool').value") == "none", "Escape did not turn target drawing off"
            js("document.querySelector('#cameraTargetLabel').focus();document.querySelector('#cameraTargetTool').value='none';document.querySelector('#cameraTargetLabel').dispatchEvent(new KeyboardEvent('keydown',{key:'q',bubbles:true}))")
            assert js("document.querySelector('#cameraTargetTool').value") == "none", "Shortcut fired while editing an input"
            js("document.querySelector('#cameraTargetLabel').blur()")
            report["steps"].append("camera-target-keyboard-shortcuts")
            capture_shortcut = js("""(()=>{const b=document.querySelector('#takeSnapshot');window.__oldCaptureClick=b.onclick;window.__captureShortcutCount=0;b.onclick=()=>window.__captureShortcutCount++;b.disabled=false;document.dispatchEvent(new KeyboardEvent('keydown',{key:'s',bubbles:true}));return window.__captureShortcutCount})()""")
            assert capture_shortcut == 1, "S did not trigger one camera capture"
            blocked_while_typing = js("""(()=>{const input=document.querySelector('#cameraTargetLabel');input.focus();input.dispatchEvent(new KeyboardEvent('keydown',{key:'s',bubbles:true}));input.blur();const count=window.__captureShortcutCount,b=document.querySelector('#takeSnapshot');b.onclick=window.__oldCaptureClick;b.disabled=true;return count})()""")
            assert blocked_while_typing == 1, "S triggered camera capture while editing an input"
            assert js("document.querySelector('#takeSnapshot').innerText.includes('S')"), "Capture button does not show the shortcut"
            keycap_style = json.loads(js("JSON.stringify((()=>{const e=document.querySelector('#takeSnapshot kbd'),s=getComputedStyle(e);return {fontSize:parseFloat(s.fontSize),weight:parseInt(s.fontWeight),width:parseFloat(s.minWidth),height:parseFloat(s.height),color:s.color,background:s.backgroundColor}})())"))
            assert keycap_style["fontSize"] >= 12 and keycap_style["weight"] >= 700, keycap_style
            assert keycap_style["width"] >= 21 and keycap_style["height"] >= 20, keycap_style
            assert keycap_style["color"] != keycap_style["background"], keycap_style
            report["steps"].append("camera-capture-s-shortcut")
            for width, height in SIZES:
                view.resize(width, height)
                capture(f"camera-zero-assets-{width}x{height}", "camera", 0)
            view.resize(1440, 900)
            click("#toggleSourceInspector")
            wait("document.querySelector('.source-workspace').classList.contains('inspector-open') && document.querySelector('#startCamera').getClientRects().length>0 && document.querySelector('#takeSnapshot').getClientRects().length>0")
            assert js("document.querySelector('#cameraSettingsInspector').classList.contains('active-inspector')"), "Camera settings drawer did not open"
            assert not js("document.querySelector('#imageToolsInspector').getClientRects().length"), "Image tools leaked into camera settings"
            view.grab().save(str(output / "camera-settings-drawer-open-1440x900.png"))
            click("#closeSourceInspector")
            wait("!document.querySelector('.source-workspace').classList.contains('inspector-open')")
            click("#toggleImageTools")
            wait("document.querySelector('#imageToolsInspector').getClientRects().length>0 && document.querySelector('.target-shortcut-guide').getClientRects().length>0 && document.querySelector('#processingMode').getClientRects().length>0")
            assert js("document.querySelector('#imageToolsInspector').classList.contains('active-inspector')"), "Image tools drawer did not open"
            assert not js("document.querySelector('#cameraSettingsInspector').getClientRects().length"), "Camera settings leaked into image tools"
            target_layout = json.loads(js("""JSON.stringify((()=>{const b=e=>e.getBoundingClientRect().toJSON(),section=document.querySelector('.camera-target-section'),hints=[...document.querySelectorAll('[data-target-hint]')].map(b),options=[...document.querySelectorAll('.target-option-row label')].map(b),field=document.querySelector('.target-field-row'),footer=document.querySelector('.target-footer-row');return {section:b(section),hints,options,field:b(field),footer:b(footer)}})())"""))
            assert target_layout["section"]["height"] < 245, target_layout
            assert max(item["top"] for item in target_layout["hints"])-min(item["top"] for item in target_layout["hints"]) < 1, target_layout
            assert abs(target_layout["options"][0]["top"]-target_layout["options"][1]["top"]) < 1, target_layout
            assert target_layout["footer"]["top"] >= target_layout["field"]["bottom"], target_layout
            view.grab().save(str(output / "image-tools-drawer-open-1440x900.png"))
            click("#closeSourceInspector")
            wait("!document.querySelector('.source-workspace').classList.contains('inspector-open')")
            report["steps"].append("settings-drawer-hidden-open-close")
            view.resize(1440, 900)
            select_source("files")
            assert js("document.querySelector('#manualImport').tagName==='DETAILS' && document.querySelector('#manualImport').contains(document.querySelector('#importPaths'))")
            js("document.querySelector('#manualImport').open=true")
            fill("#importPaths", str(source))
            select_source("video")
            video_path = str(root / "保留輸入測試.mp4")
            fill("#videoPath", video_path)
            fill("#videoInterval", "0.8")
            for name in ("screen", "merge", "camera", "files", "video", "files"):
                select_source(name)
            assert js("document.querySelector('#importPaths').value") == str(source), "File path lost on source switch"
            assert js("document.querySelector('#videoPath').value") == video_path, "Video path lost on source switch"
            assert js("document.querySelector('#videoInterval').value") == "0.8", "Video interval lost on source switch"
            assert js("document.querySelector('#manualImport').open"), "Expanded path section lost on source switch"
            click("#importFiles")
            wait(f"document.querySelector('#acquiredAssets').querySelectorAll('img').length==={THUMBNAIL_COUNT}", seconds=60)
            wait("!window.workbenchState().busy && !document.querySelector('#importFiles').disabled")
            wait("(()=>{const i=document.querySelector('#acquiredAssets img');return i&&i.complete&&i.naturalWidth>0})()")
            assert str(js("document.querySelector('#acquireTotal').textContent")) == str(IMAGE_COUNT), "Image total did not update"
            project = service.store.list_projects()[0]
            assert project["stats"]["total"] == IMAGE_COUNT, project
            report["project_images"] = IMAGE_COUNT
            report["steps"].append("source-switch-input-retention-and-real-import")
            # Capture the settled workspace after the transient success toast closes.
            wait("document.querySelector('#toast').hidden", seconds=12)
            view.resize(1440, 900)
            select_source("camera")
            assert js("document.querySelector('#togglePreviewFullscreen').getClientRects().length>0"), "Preview expand control is missing"
            click("#togglePreviewFullscreen")
            wait("document.querySelector('#acquire').classList.contains('preview-expanded-layout')")
            expanded = json.loads(js("""JSON.stringify((()=>{const b=e=>e.getBoundingClientRect().toJSON(),w=document.querySelector('.source-workspace'),t=document.querySelector('.acquired-tray'),a=document.querySelector('#acquiredAssets'),c=document.querySelector('.compact-camera-capture');return {workspace:b(w),tray:b(t),capture:b(c),header:b(document.querySelector('.app-header')),phase:b(document.querySelector('.phasebar')),assetDirection:getComputedStyle(a).flexDirection,assetOverflow:getComputedStyle(a).overflowX,pressed:document.querySelector('#togglePreviewFullscreen').getAttribute('aria-pressed')}})())"""))
            assert expanded["workspace"]["width"] > 1350, expanded
            assert expanded["tray"]["top"] >= expanded["workspace"]["bottom"], expanded
            assert expanded["capture"]["height"] < 75, expanded
            assert expanded["header"]["height"] > 0 and expanded["phase"]["height"] > 0, expanded
            assert expanded["assetDirection"] == "row" and expanded["assetOverflow"] in ("auto", "scroll"), expanded
            assert expanded["pressed"] == "true", expanded
            QTest.qWait(500)
            view.grab().save(str(output / "camera-preview-expanded-1440x900.png"))
            js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))")
            wait("!document.querySelector('#acquire').classList.contains('preview-expanded-layout')")
            assert js("document.querySelector('#togglePreviewFullscreen').getAttribute('aria-pressed')") == "false"
            report["steps"].append("camera-preview-expanded-reflow-and-escape")
            for width, height in SIZES:
                view.resize(width, height)
                QTest.qWait(200)
                for name in SOURCES:
                    select_source(name)
                    suffix = "empty" if name == "camera" else "imported" if name == "files" else "source"
                    capture(f"{name}-{suffix}-{width}x{height}", name, THUMBNAIL_COUNT)
            view.resize(1910, 992)
            view.setZoomFactor(1.25)
            for name in SOURCES:
                select_source(name)
                capture(f"{name}-52-assets-1910x992-zoom125", name, THUMBNAIL_COUNT)
            view.setZoomFactor(1.0)
            assert not report["violations"], f"{len(report['violations'])} layout violations; see report.json geometry"
            view.resize(1440, 900)
            select_source("files")
            thumb = js("(()=>{const img=document.querySelector('#acquiredAssets img');const e=img?.closest('button,[role=button],a');return !!e})()")
            assert thumb, "Acquired image is not an actionable thumbnail"
            # Reach the last thumbnail through the tray's scrollbar, then use a
            # native pointer event on its visible center rather than DOM click.
            js("document.querySelector('#acquiredAssets').scrollTop=document.querySelector('#acquiredAssets').scrollHeight")
            QTest.qWait(120)
            point = json.loads(js("JSON.stringify((()=>{const r=document.querySelector('#acquiredAssets .acquired-item:last-child').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})())"))
            from PySide6.QtCore import QPoint, Qt
            QTest.mouseClick(view.focusProxy() or view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                QPoint(round(point["x"]), round(point["y"])))
            wait("!document.querySelector('#annotate').hidden && !window.workbenchState().transitioning")
            wait("!document.querySelector('#imageStage').hidden && document.querySelector('#assetImage').complete")
            assert js("!!window.workbenchState().assetId"), "Thumbnail did not open an asset"
            assert service.store.get_project(project["id"])["stats"]["total"] == IMAGE_COUNT
            report["steps"].append("acquired-thumbnail-opens-annotation")
            assert not page.errors, page.errors
            report["success"] = True
            print("ACQUISITION_LAYOUT_OK", flush=True)
        except Exception as error:
            report["error"] = str(error)
            view.grab().save(str(output / "failure.png"))
            try:
                (output / "failure-state.txt").write_text(str(js("document.body.innerText")), encoding="utf-8")
            except Exception:
                pass
            raise
        finally:
            report["page_errors"] = page.errors
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            # Navigate away before deleting the temporary project. Chromium can
            # otherwise retain the last decoded project image handle on Windows.
            view.setUrl(QUrl("about:blank"))
            QTest.qWait(250)
            view.close()
            page.deleteLater()
            QTest.qWait(250)
            service.close()


if __name__ == "__main__":
    main()
