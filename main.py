from __future__ import annotations
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Vision Workbench — local image data pipeline")
    parser.add_argument("--serve", action="store_true", help="Run loopback UI for development and verification")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = (args.data_root or Path(__file__).resolve().parent / "data").resolve()
    (root / "logs").mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[RotatingFileHandler(root/"logs"/"workbench.log",maxBytes=5_000_000,backupCount=3,encoding="utf-8")])
    if args.serve:
        from workbench.server import WorkbenchService
        service = WorkbenchService(root,args.port).start()
        print(service.entry_url,flush=True)
        try:
            while True:
                time.sleep(.5)
        except KeyboardInterrupt:
            pass
        finally:
            service.close()
        return 0
    from workbench.desktop import run_desktop
    return run_desktop(root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Application startup failed")
        raise
