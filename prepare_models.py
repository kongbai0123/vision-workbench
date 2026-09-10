"""Explicit model preparation; inference itself never requires network access."""
from pathlib import Path
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only",action="store_true")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    source = Path(snapshot_download("facebook/sam2.1-hiera-tiny",
        revision="de431c4043854a71d8101e17995dfe596bf101a5",local_files_only=args.local_only))
    target = Path(__file__).resolve().parent / "models" / "sam2.1-hiera-tiny"
    target.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,target,dirs_exist_ok=True,symlinks=False)
    if not (target / "config.json").is_file() or not list(target.glob("*.safetensors")):
        raise RuntimeError("Model snapshot incomplete")
    print(f"Local model prepared: {target}")


if __name__ == "__main__":
    main()
