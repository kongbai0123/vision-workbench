"""Exercise review selection and recoverable actions in isolated Qt Chromium."""
import json
import tempfile
import time
from pathlib import Path
from desktop_workflow import QApplication, QWebEngineView, Page, QUrl, QEventLoop, QTimer, QTest, Image, WorkbenchService


def main():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        service = WorkbenchService(root / 'data').start()
        pid = service.store.create_project('Review UI')['id']
        for i in range(3):
            image = root / f'image{i}.png'
            Image.new('RGB', (320, 240), (30+i*60, 90, 120)).save(image)
            service.store.add_assets(pid, [{'path': image, 'shapes': []}])
        view = QWebEngineView(); page = Page(view); view.setPage(page)
        view.resize(1440, 900); view.show(); view.setUrl(QUrl(service.url))
        def js(script):
            result=[]; loop=QEventLoop()
            page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit); loop.exec()
            return result[0] if result else None
        def wait(expression):
            end=time.monotonic()+30
            while time.monotonic()<end:
                if js(f'Boolean({expression})'): return
                QTest.qWait(80)
            raise AssertionError(expression + '\n' + str(js('document.body.innerText.slice(-2000)')))
        def click(selector):
            wait("!document.querySelector('#home').disabled")
            js(f'document.querySelector({json.dumps(selector)}).click()')
        try:
            wait("document.querySelector('.project-card-open')")
            click('.project-card-open')
            click('[data-stage="review"]')
            wait("document.querySelectorAll('.review-card').length===3")
            click('.review-card-media img')
            assert js("!document.querySelector('#review').hidden && document.querySelectorAll('.review-card.selected').length===1")
            click('.review-card-footer button:last-child')
            wait("!document.querySelector('#annotate').hidden")
            click('[data-stage="review"]')
            wait("!document.querySelector('#review').hidden && document.querySelectorAll('.review-card.selected').length===1")
            click('#reviewReject');wait("document.querySelector('#formDialog').open")
            js("document.querySelector('#confirmDialog').click()")
            wait("!document.querySelector('#formDialog').open && document.querySelectorAll('.review-card').length===2")
            click('.review-card-media img');click('#reviewTrash')
            wait("document.querySelector('#formDialog').open")
            js("document.querySelector('#confirmDialog').click()")
            wait("!document.querySelector('#formDialog').open && document.querySelectorAll('.review-card').length===1")
            click('#reviewRestore');wait("document.querySelector('#dialogBody input[type=checkbox]')")
            js("document.querySelector('#dialogBody input').click();document.querySelector('#confirmDialog').click()")
            wait("!document.querySelector('#formDialog').open && document.querySelectorAll('.review-card').length===2")
            assert not js('document.documentElement.scrollWidth>innerWidth+2')
            QTest.qWait(300)
            output=Path(__file__).resolve().parents[1]/'qa-output'/'desktop'
            output.mkdir(parents=True,exist_ok=True);view.grab().save(str(output/'review-workflow.png'))
            click('[data-stage="acquire"]')
            js(f"document.querySelector('#importPaths').value={json.dumps(str(root/'image0.png'))}")
            click('#importFiles');wait("document.querySelector('#formDialog').open && document.querySelector('#dialogBody input[type=checkbox]')")
            js("document.querySelector('#dialogBody input[type=checkbox]').click();document.querySelector('#confirmDialog').click()")
            QTest.qWait(150)
            assert js("document.querySelector('#formDialog').open")
            js("document.querySelector('#cancelDialog').click()")
            wait("!document.querySelector('#formDialog').open")
            click('#importFiles');wait("document.querySelector('#formDialog').open && document.querySelector('#dialogBody input[type=checkbox]')")
            js("document.querySelector('#confirmDialog').click()")
            wait("!document.querySelector('#formDialog').open && !document.querySelector('#home').disabled")
            assert not page.errors, page.errors
            print('PASS: card selection, edit/return, exclusion, trash/restore, layout and import preview cancellation',flush=True)
        finally:
            view.close();service.close();QTest.qWait(100)


if __name__ == '__main__':
    main()
