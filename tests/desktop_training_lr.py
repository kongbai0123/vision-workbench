"""Focused Qt rendering check for merged LR curves; no training is launched."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
HARNESS = """<!doctype html><meta charset=utf-8>
<link rel=stylesheet href=training.css><style>body{margin:20px;background:#10191f;color:#c4d9e0;font-family:sans-serif}</style>
<div id=plots></div><script type=module>
import {createTrainingCharts,metricDescriptors} from './training-charts.mjs';
const runs=[1,2].map(n=>({run_id:`R00${n}`,engine:'yolo26n_seg',appearanceIndex:n-1,config:{epochs:50,learning_rate:.001}}));
window.reports=new Map(runs.map((run,n)=>[run.run_id,{run,metrics:[1,2,3].map(epoch=>({epoch,'train/loss':1/epoch,
'val/mask_map50_95':.4+epoch*.1,'train/learning_rate':.001/epoch,'lr/group_0':.001/epoch,'lr/group_1':.001/epoch}))}]));
window.draw=(selected=runs)=>{window.chart?.destroy();window.chart=createTrainingCharts(document.querySelector('#plots'),
{runs:selected,domainRuns:runs,reports,metricKeys:metricDescriptors(selected,reports).map(item=>item.key)});};
window.draw();window.ready=true;
</script>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/smoke":
            body = HARNESS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *_args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT / "web")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.resize(1100, 1100)
    view.show()

    def js(source):
        loop, result = QEventLoop(), []
        view.page().runJavaScript(source, lambda value: (result.append(value), loop.quit()))
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        assert result, "JavaScript callback timed out"
        return result[0]

    def wait(source):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if js(source):
                return
            app.processEvents()
            time.sleep(.03)
        raise AssertionError(source)

    try:
        view.setUrl(QUrl(f"http://127.0.0.1:{server.server_port}/smoke"))
        wait("window.ready===true")
        assert js("document.querySelectorAll('.training-chart-host>svg').length") == 3
        selector = "document.querySelector('[data-metric-key=\"train/learning_rate\"]')"
        assert js(selector + ".querySelectorAll('.training-chart-host polyline').length") == 2
        assert js(selector + ".innerText.includes('參數組 1')")
        js("reports.get('R002').metrics[1]['lr/group_1']=.0009;draw()")
        assert js(selector + ".querySelectorAll('.training-chart-host polyline').length") == 4
        assert js(selector + ".querySelector('.training-chart-host svg').dataset.xMax") == "50"
        js(selector + ".querySelector('[data-chart-hit]').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
        assert js(selector + ".querySelector('.training-chart-hover').innerText.includes('參數組 1')")
        assert js("document.querySelectorAll('input[type=checkbox]').length") == 0
        view.resize(560, 1100)
        wait("document.documentElement.clientWidth>0&&document.documentElement.clientWidth<=560")
        assert js("document.documentElement.scrollWidth<=document.documentElement.clientWidth")
        print(json.dumps({"lr_panels": 1, "same_group_curves": 2, "different_group_curves": 4,
                          "other_metric_panels": 2, "narrow_layout": "passed"}))
    finally:
        view.close()
        server.shutdown()


if __name__ == "__main__":
    main()
