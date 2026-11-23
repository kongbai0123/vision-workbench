"""Compare Gemma with Qwen on unchanged MM candidates inside the offline worker."""

import argparse
from collections import Counter
from contextlib import ExitStack
import faulthandler
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from PIL import Image

if __package__:
    from .annotation_assist_gguf import NativeVision, RUNTIME_COMMIT, probabilities_from_logits
    from .annotation_assist_grounding import HEALTH_EXAMPLES
    from .annotation_assist_worker import (
        QwenJudge, RESULT_SCHEMA, emit, file_hash, frame_image, proposal_view, read_manifest, select_vote,
        valid_box, write_result,
    )
else:
    from annotation_assist_gguf import NativeVision, RUNTIME_COMMIT, probabilities_from_logits
    from annotation_assist_grounding import HEALTH_EXAMPLES
    from annotation_assist_worker import (
        QwenJudge, RESULT_SCHEMA, emit, file_hash, frame_image, proposal_view, read_manifest, select_vote,
        valid_box, write_result,
    )


SCHEMA = "vision_workbench.annotation_assist_gemma.v1"
MAX_CASES = 96
MODEL_REVISION = "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"
MODEL_FILES = {
    "gemma-4-E4B_q4_0-it.gguf": "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
    "gemma-4-E4B-it-mmproj.gguf": "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",
}
RUNTIME_FILES = {
    "llama.dll": "98451bd70daaf956902b000210568820f09afb4c5e74e46531d0ba6cfe4c2b15",
    "mtmd.dll": "50bd160608436d15e6a5dd2af5912a33f3d276a6714107c66468b35500625ac4",
    "ggml.dll": "891f0674a1aaf8f4cc09e5f5b73b3d254d65107f6e5c5e598426762488a93c7e",
    "ggml-base.dll": "8623985ac55d5fd3d0b5e7c41941166e5ff0d247d7a80eea21509163d3534af7",
    "ggml-cuda.dll": "43613a4001d2a87ae6f4861ade0fd3877e2a1d9e151f10b56eccbbeccb0023ba",
    "ggml-cpu-x64.dll": "552202da29c3356a5725cfb74afcdf7e9c8e7b2601b08604ecec791aa266c1fc",
    "cublas64_13.dll": "db5a1d5a0ed0ef126e005fc7449d3584353aaeb8739a18d716fe74be1254a3da",
    "cublasLt64_13.dll": "ca478e1ea932f883ef4fc284f8f33d910079df4e77f9a2ec38d9b4387795f96f",
    "cudart64_13.dll": "16ab35d8150b6105b7e49da264d360db4ece090157ad48d226c1dc7254136008",
    "libomp.dll": "1e4222f195d84e7355109d778e751dfe3c085e42c947f7fe3d7853181329dd40",
}
REASONS = {
    "accepted": "Gemma 兩次均判為符合類別與框界。",
    "bad_extent": "Gemma 認為這個位置有目標，框界或實例數需要確認。",
    "different_object": "Gemma 認為候選框內是其他物件或類別。",
    "uncertain_review": "Gemma 的兩次裁決分歧，或選項證據不足。",
}
CONTROL_NAMES = {"blank": "空白圖", "wrong_location": "錯位框", "wrong_class": "錯誤類別", "positive": "已知正例"}


def gemma_template(text, marker):
    return b"<|turn>user\n" + marker + text.encode("utf-8") + b"<turn|>\n<|turn>model\n"


def checked_assets(model_home):
    model_dir = model_home / "Gemma-4-E4B-it-QAT-Q4"
    runtime = model_home / "_runtimes" / "llama.cpp-b12541-cuda13.5"
    for directory, pins in ((model_dir, MODEL_FILES), (runtime, RUNTIME_FILES)):
        for name, expected in pins.items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Gemma 本機檔案雜湊不符：{name}")
    return runtime, *(model_dir / name for name in MODEL_FILES)


def choice_tokens(model):
    token_choices = {}
    for index, letter in enumerate("ABCD"):
        for form in (letter, " " + letter, "\n" + letter):
            ids = model.encode(form)
            if len(ids) == 1:
                if token_choices.setdefault(ids[0], index) != index:
                    raise ValueError("Gemma 的不同裁決共用了同一個答案 token。")
        if index not in token_choices.values():
            raise ValueError(f"Gemma 無法以單一 token 表達選項 {letter}。")
    return list(token_choices), list(token_choices.values())


def case_priority(candidate):
    score = candidate.get("score")
    score = score if type(score) in (float, int) and isfinite(score) else -1
    return bool(candidate.get("votes")), score


def collect_cases(manifest, inference):
    frames = {row["id"]: row for row in manifest["frames"]}
    definitions = {row["label"]: row for row in inference["method"]["classes"]}
    if set(definitions) != {row["label"] for row in manifest["config"]["classes"]}:
        raise ValueError("Gemma 的類別定義與原批次不符。")
    by_kind = {verdict: [] for verdict in ("accepted", "unresolved", "rejected")}
    seen_frames = set()
    eligible = 0
    for result in inference["frames"]:
        frame_id = result["id"]
        frame = frames.get(frame_id)
        if (frame is None or frame_id in seen_frames
                or result["image_sha256"] != frame["image_sha256"]):
            raise ValueError("Gemma 的影像識別或雜湊與原批次不符。")
        seen_frames.add(frame_id)
        seen_candidates = set()
        shortlisted = {}
        for candidate in result["candidates"]:
            identifier = candidate["id"]
            if not isinstance(identifier, str) or not identifier or identifier in seen_candidates:
                raise ValueError("Gemma 收到重複或無效的候選識別碼。")
            seen_candidates.add(identifier)
            verdict = candidate["verdict"]
            if (verdict not in by_kind or candidate["label"] not in definitions
                    or not valid_box(candidate["xyxy"], frame["width"], frame["height"])):
                continue
            eligible += 1
            current = shortlisted.get(verdict)
            if current is None or case_priority(candidate) > case_priority(current):
                shortlisted[verdict] = candidate
        for verdict, candidate in shortlisted.items():
            by_kind[verdict].append((frame, candidate, definitions[candidate["label"]]))
    if seen_frames != frames.keys():
        raise ValueError("Gemma 的影像清單與原始推論不完整。")
    # Alternate verdicts so a large batch still includes hard cases and rejected controls.
    cases = []
    for index in range(max(map(len, by_kind.values()))):
        for group in by_kind.values():
            if index < len(group):
                cases.append(group[index])
                if len(cases) == MAX_CASES:
                    return cases, eligible
    return cases, eligible


def observe_panel(model, tokens, image, box, definition, settings):
    panel = proposal_view(image, box)
    judge = object.__new__(QwenJudge)
    label = definition["label"]
    judge.classes = {label: definition}
    pairs, measurements = [], []
    for reverse in (False, True):
        logits, info = model.vote(panel, judge.review_prompt(label, reverse))
        pairs.append(probabilities_from_logits(logits, *tokens))
        measurements.append(info)
    decision = select_vote(pairs, settings)
    for vote, metrics in zip(decision["votes"], measurements):
        vote.update(metrics)
    return dict(decision, reason=REASONS[decision["reason_code"]],
                panel_sha256=sha256(panel.tobytes()).hexdigest())


def health_status(controls):
    failed = [entry["kind"] for entry in controls
              if entry["verdict"] == ("rejected" if entry["expected"] == "accepted" else "accepted")]
    if failed:
        return {"status": "failed", "reason": "健全性檢查未通過：" + "、".join(CONTROL_NAMES[name] for name in failed),
                "controls": controls}
    if ({entry["kind"] for entry in controls} != CONTROL_NAMES.keys()
            or any(entry["verdict"] != entry["expected"] for entry in controls)):
        return {"status": "inconclusive", "reason": "部分健全性控制的裁決不明確。", "controls": controls}
    return {"status": "passed", "reason": "Gemma 通過空白、錯位、錯類與已知正例檢查。", "controls": controls}


def run_health_checks(model, tokens, model_home, settings):
    fixture = HEALTH_EXAMPLES[0]
    path = model_home / "MM-Grounding-DINO-Tiny" / "health" / fixture["file"]
    if not path.is_file() or file_hash(path) != fixture["sha256"]:
        return {"status": "not_run", "reason": "缺少已核對的 Gemma 健全性對照影像。", "controls": []}
    with Image.open(path) as source:
        known_image = source.convert("RGB")
    box = fixture["xyxy"]
    if not valid_box(box, known_image.width, known_image.height):
        return {"status": "not_run", "reason": "已知正例框超出對照影像範圍。", "controls": []}
    blank = Image.new("RGB", known_image.size)
    wrong_location = Image.new("RGB", (known_image.width * 2, known_image.height))
    wrong_location.paste(known_image, (known_image.width, 0))
    tests = (("blank", blank, "car", "rejected"), ("wrong_location", wrong_location, "car", "rejected"),
             ("wrong_class", known_image, "screwdriver", "rejected"), ("positive", known_image, "car", "accepted"))
    controls = []
    for kind, image, label, expected in tests:
        definition = {"label": label, "prompt": label, "description": "Cover one visible instance of the named object."}
        decision = observe_panel(model, tokens, image, box, definition, settings)
        controls.append(dict(decision, kind=kind, label=label, expected=expected, xyxy=box))
    return health_status(controls)


def observe_cases(model, tokens, cases, root, settings):
    rows = []
    last_id, image = None, None
    for frame, candidate, definition in cases:
        if frame["id"] != last_id:
            last_id = frame["id"]
            image = frame_image(root, frame)
        result = observe_panel(model, tokens, image, candidate["xyxy"], definition, settings)
        votes = candidate.get("votes", [])
        qwen_verdict = None
        if len(votes) == 2:
            pairs = [([vote["probabilities"][letter] for letter in "ABCD"], vote["allowed_token_mass"]) for vote in votes]
            qwen_verdict = select_vote(pairs, settings)["verdict"]
        comparison = ("qwen_not_run" if qwen_verdict is None
                      else "agreement" if qwen_verdict == result["verdict"] else "disagreement")
        rows.append(dict(result, asset_id=frame["id"], image_sha256=frame["image_sha256"],
                         candidate_id=candidate["id"], label=candidate["label"], xyxy=candidate["xyxy"],
                         qwen_verdict=qwen_verdict, qwen_comparison=comparison))
        emit("gemma", completed=len(rows), total=len(cases), frame_id=frame["id"], candidate_id=candidate["id"])
    return rows


def observe(manifest_path, inference_path, model_home, output):
    manifest_path = manifest_path.resolve(strict=True)
    manifest, settings, manifest_hash = read_manifest(manifest_path)
    original = inference_path.read_bytes()
    inference_hash = sha256(original).hexdigest()
    inference = json.loads(original)
    if (inference.get("schema") != RESULT_SCHEMA or inference.get("status") != "complete"
            or inference.get("job_id") != manifest["job_id"] or inference.get("manifest_sha256") != manifest_hash):
        raise ValueError("Gemma 的觀察輸入與原批次清單不符。")
    result = {"schema": SCHEMA, "observation_only": True, "job_id": manifest["job_id"],
              "manifest_sha256": manifest_hash, "inference_sha256": inference_hash, "rows": []}
    cases, eligible = collect_cases(manifest, inference)
    if not cases:
        result.update(health={"status": "not_run", "reason": "這批沒有有效的候選框。", "controls": []},
                      summary={"observed": 0, "eligible": eligible, "budget": MAX_CASES})
        write_result(output, result)
        return result
    started = perf_counter()
    emit("gemma", completed=0, total=len(cases))
    runtime, model_path, vision_path = checked_assets(model_home)
    with ExitStack() as stack:
        model = NativeVision(stack, runtime, model_path, vision_path, chat_prompt=gemma_template)
        tokens = choice_tokens(model)
        health = run_health_checks(model, tokens, model_home, settings)
        rows = observe_cases(model, tokens, cases, manifest_path.parent, settings)
        model.memory.append(model.memory_snapshot("after_reviews"))
        timing = {"load_seconds": round(model.load_seconds, 3), "elapsed_seconds": round(perf_counter() - started, 3),
                  "device": model.gpu_name, "memory": model.memory}
    if file_hash(manifest_path) != manifest_hash or file_hash(inference_path) != inference_hash:
        raise ValueError("Gemma 觀察期間，原批次或推論檔案已變更。")
    comparisons = Counter(row["qwen_comparison"] for row in rows)
    result.update(health=health, rows=rows, timing=timing,
                  model={"name": "Gemma 4 E4B Instruct QAT Q4", "revision": MODEL_REVISION, "gguf": MODEL_FILES,
                         "llama_cpp_revision": RUNTIME_COMMIT, "runtime_files": RUNTIME_FILES},
                  summary={"observed": len(rows), "eligible": eligible, "budget": MAX_CASES, **comparisons})
    write_result(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "inference", "model-home", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    # The isolated process has its own log, including Python stacks on a native abort.
    faulthandler.enable()
    try:
        observe(args.manifest, args.inference, args.model_home, args.output)
    except Exception as exc:
        print(json.dumps({"event": "error", "message": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
