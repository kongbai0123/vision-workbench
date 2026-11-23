"""Measure SAM 2.1 mask bounds against frozen MM candidate boxes."""

from hashlib import file_digest
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter

if __package__:
    from .annotation_assist_grounding import overlap
else:
    from annotation_assist_grounding import overlap


SAM_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM_FILES = {
    "config.json": (5695, "860aff9751b139d83a4ad7df1e5535416fded533e0ead02625edbefcb9953cce"),
    "model.safetensors": (155908064, "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"),
    "preprocessor_config.json": (683, "6ebf229ee259368ce4a8d4f2fe893a72b053023710853e257253939e601f583d"),
    "processor_config.json": (95, "f8a68e865cfad115c1c2763f3d93eca7b1c622da06da2a9273eb437fb2389b6d"),
}
MINIMUM_BOUNDARY_IOU = 0.75
MINIMUM_MASK_SCORE = 0.60
MAX_BOXES_PER_FRAME = 6
REVIEW_REASONS = {
    "sam_boundary_disagreement": "SAM 2.1 遮罩外接框與 MM 原框的 IoU 低於 0.75，請比對物件邊界。",
    "sam_boundary_unavailable": "SAM 2.1 未取得可信的遮罩邊界，請人工確認原框範圍。",
    "sam_boundary_budget": "本張已用完 SAM 2.1 檢查額度，此建議框的邊界待人工確認。",
}


def verify_sam_files(directory):
    directory = Path(directory).resolve(strict=True)
    for name, (size, checksum) in SAM_FILES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"SAM 2.1 本機模型版本或檔案大小不符：{name}")
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != checksum:
                raise ValueError(f"SAM 2.1 本機模型雜湊不符：{name}")
    return directory


class SamBoundaryEvidence:
    """Local box-prompt segmentation with one image embedding per frame."""

    def __init__(self, directory):
        self.directory = verify_sam_files(directory)
        self.provenance = {
            "model": "facebook/sam2.1-hiera-tiny", "revision": SAM_REVISION,
            "files": {name: digest for name, (_, digest) in SAM_FILES.items()},
            "minimum_boundary_iou": MINIMUM_BOUNDARY_IOU, "minimum_mask_score": MINIMUM_MASK_SCORE,
            "max_boxes_per_frame": MAX_BOXES_PER_FRAME, "mode": "box_prompt_boundary_evidence",
        }
        self.torch = self.processor = self.model = None
        self.image_key, self.embeddings = None, None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import Sam2Model, Sam2Processor

        processor = Sam2Processor.from_pretrained(self.directory, local_files_only=True)
        model = Sam2Model.from_pretrained(self.directory, local_files_only=True, use_safetensors=True)
        self.model = model.to(device="cuda:0", dtype=torch.float16).eval()
        self.torch, self.processor = torch, processor

    def measure(self, image, box, image_hash):
        self._load()
        width, height = image.size
        cached = self.image_key == (image_hash, image.size) and self.embeddings is not None
        source = {"original_sizes": [[height, width]]} if cached else {"images": image}
        inputs = self.processor(**source, input_boxes=[[box]], return_tensors="pt")
        original_sizes = inputs.pop("original_sizes")
        pixels = inputs.pop("pixel_values", None)
        torch = self.torch
        if not cached:
            self.image_key, self.embeddings = None, None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if not cached:
                self.embeddings = self.model.get_image_embeddings(pixels.to("cuda:0"))
                self.image_key = image_hash, image.size
            output = self.model(**inputs.to("cuda:0"), image_embeddings=self.embeddings, multimask_output=True)
        masks = self.processor.post_process_masks(output.pred_masks.detach().float().cpu(), original_sizes)[0]
        if tuple(masks.shape[-2:]) != (height, width):
            raise RuntimeError("SAM 2.1 遮罩尺寸與原圖不符。")
        scores = output.iou_scores.detach().float().cpu().reshape(-1).tolist()
        x1, y1, x2, y2 = box
        masks = masks.reshape(-1, height, width)
        if len(scores) != len(masks):
            raise RuntimeError("SAM 2.1 遮罩與品質分數數量不同。")
        for index in sorted(range(len(scores)), key=lambda i: scores[i] if isfinite(scores[i]) else -1, reverse=True):
            mask, score = masks[index], scores[index]
            if not isfinite(score) or not mask[floor(y1):ceil(y2), floor(x1):ceil(x2)].any():
                continue
            rows, columns = mask.any(dim=1).nonzero().flatten(), mask.any(dim=0).nonzero().flatten()
            return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1], score
        return None, None

    def close(self):
        self.image_key = self.embeddings = self.model = self.processor = None


def _evidence(status, *, code=None, reason=None, box=None, iou=None, score=None, seconds=None):
    return {"status": status, "reason_code": code, "reason": reason or REVIEW_REASONS.get(code),
            "mask_box_xyxy": box, "iou": iou, "mask_score": score, "seconds": seconds}


def _measure(engine, image, candidate, image_hash):
    started = perf_counter()
    try:
        box, score = engine.measure(image, candidate["xyxy"], image_hash)
    except (RuntimeError, ValueError) as exc:
        observation = _evidence("unavailable", code="sam_boundary_unavailable")
        observation["diagnostic"] = str(exc)[:300]
    else:
        iou = overlap(candidate["xyxy"], box) if box else None
        if iou is None or score is None or not isfinite(score) or score < MINIMUM_MASK_SCORE:
            observation = _evidence("unavailable", code="sam_boundary_unavailable", box=box, iou=iou,
                                    score=score if score is not None and isfinite(score) else None)
        elif iou < MINIMUM_BOUNDARY_IOU:
            observation = _evidence("mismatch", code="sam_boundary_disagreement", box=box, iou=iou, score=score)
        else:
            observation = _evidence("aligned", reason="SAM 2.1 遮罩外接框與 MM 原框的 IoU 達 0.75。",
                                    box=box, iou=iou, score=score)
    observation["seconds"] = round(perf_counter() - started, 6)
    return observation


def check_sam_frame(engine, image, frame):
    """Add evidence and review reasons, keeping MM geometry and Qwen decisions intact."""
    eligible = sorted((candidate for candidate in frame["candidates"]
                       if candidate["verdict"] in {"accepted", "unresolved"} and candidate["xyxy"] is not None),
                      key=lambda candidate: (candidate["verdict"] != "accepted", -candidate["score"], candidate["id"]))
    shortlist = {candidate["id"] for candidate in eligible[:MAX_BOXES_PER_FRAME]}
    eligible_ids = {candidate["id"] for candidate in eligible}
    for candidate in frame["candidates"]:
        ident = candidate["id"]
        if ident in shortlist:
            observation = _measure(engine, image, candidate, frame["image_sha256"])
        elif ident in eligible_ids:
            observation = _evidence("skipped_budget", code="sam_boundary_budget")
        else:
            observation = _evidence("not_eligible", reason="本輪只檢查 Qwen 採用或待確認且有有效座標的候選。")
        candidate["sam"] = observation
        if observation["reason_code"] and (observation["status"] != "skipped_budget" or candidate["verdict"] == "accepted"):
            frame["exceptions"].append({"code": observation["reason_code"], "reason": observation["reason"],
                                        "candidate_id": ident, "label": candidate["label"], "xyxy": candidate["xyxy"]})
    if frame["exceptions"]:
        frame["status"] = "review"
