"""Local, text-conditioned proposals for the approved MM Grounding DINO checkpoints."""

from copy import deepcopy
from hashlib import file_digest
import os
from pathlib import Path

from PIL import Image


DEFAULT_CANDIDATE_LIMIT = 5
MAX_CANDIDATE_LIMIT = 100
FILES = {
    "model.safetensors": "2d9bdfd0247f167a00e00fffa1dec1ee5d5861fdfb4e27fd8cc2279af6820688",
    "config.json": "cb8ffc83fa083aebe4a72d4714c759568b4001b32e28a1850e8c2f30c1db5923",
    "preprocessor_config.json": "c17fd68afb1f124bfb87a494d409925eec4201b3487c8a694dde64d9ce7109a3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer_config.json": "03070b03f452696da3d89489fa8a4e53730ab74ed2ae466d53923ab5376ce7b5",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
HEALTH_EXAMPLES = (
    {"file": "00010bf498b64bab.jpg", "prompt": "car", "xyxy": [592.64, 287.038656, 713.6, 386.34912],
     "sha256": "2fbef90e6bae5a3dcba323ca103716df2022a32ee37cd0b2a7daf8054975d99e"},
    {"file": "000851174d48d3d9.jpg", "prompt": "person", "xyxy": [148.48, 234.50122, 270.72, 681.078038],
     "sha256": "4ec4d724a8e9bb0addc4209c77dec1e5e837ff7c5d3d3d8349a0424bae497a8a"},
    {"file": "00a601c5da2603d2.jpg", "prompt": "person", "xyxy": [209.28, 369.28, 392.319744, 960.64],
     "sha256": "b2120616c48e83e0b62945ded7099216428db5443ec2431d35542aaacdb0c403"},
)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def model_file_pins(directory):
    config = Path(directory) / "config.json"
    if config.is_file() and digest(config) == FILES["config.json"]:
        return FILES
    raise ValueError(f"DINO 本機設定檔不存在或版本不符：{config}")


def checkpoint_sha256(directory):
    return model_file_pins(directory)["model.safetensors"]


def restore_heads(model, checkpoint):
    import torch
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name in ("bbox_embed", "class_embed"):
            original = getattr(model, name)
            if len(original) != 6:
                raise RuntimeError(f"DINO 解碼器的 {name} 層數不符。")
            heads = torch.nn.ModuleList()
            for index in range(6):
                prefix = f"{name}.0." if index == 0 else f"model.decoder.{name}.{index}."
                head = deepcopy(original[0])
                parameters = {k[len(prefix):]: source.get_tensor(k) for k in keys if k.startswith(prefix)}
                head.load_state_dict(parameters, strict=True)
                if any(not torch.equal(v, parameters[k]) for k, v in head.state_dict().items()):
                    raise RuntimeError(f"DINO 第 {index + 1} 層 {name} 參數核對失敗。")
                heads.append(head)
            setattr(model, name, heads)
            setattr(model.model.decoder, name, heads)


def load_model(directory, device):
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    for name, expected in model_file_pins(directory).items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"DINO 本機檔案不存在或版本不符：{path}")
    options = {"local_files_only": True, "trust_remote_code": False}
    processor = AutoProcessor.from_pretrained(directory, **options)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        directory, use_safetensors=True, dtype=torch.float32, use_kernels=False, **options)
    restore_heads(model, directory / "model.safetensors")
    return processor, model.to(device).eval()


def validate_candidate_limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_CANDIDATE_LIMIT:
        raise ValueError(f"候選框數量必須為 1 至 {MAX_CANDIDATE_LIMIT} 的整數。")
    return value


def detect(processor, model, device, image, prompt, *, candidate_limit=DEFAULT_CANDIDATE_LIMIT):
    candidate_limit = validate_candidate_limit(candidate_limit)
    import torch
    from torchvision.ops import nms

    values = processor(images=image, text=prompt.rstrip(".") + ".", return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**values)
    result = processor.post_process_grounded_object_detection(
        output, input_ids=values.input_ids, threshold=0.0, text_threshold=0.25,
        target_sizes=[(image.height, image.width)])[0]
    boxes, scores = result["boxes"].float(), result["scores"].float()
    boxes[:, 0::2].clamp_(0, image.width)
    boxes[:, 1::2].clamp_(0, image.height)
    keep = (torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
            & (boxes[:, 2] - boxes[:, 0] >= 8) & (boxes[:, 3] - boxes[:, 1] >= 8))
    boxes, scores = boxes[keep], scores[keep]
    order = nms(boxes, scores, 0.65)[:candidate_limit]
    return [(tuple(box), score) for box, score in zip(boxes[order].tolist(), scores[order].tolist(), strict=True)]


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class GroundingEngine:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.model = None
        self.load_error = None

    def _check_examples(self, processor, model, device):
        folder = self.directory / "health"
        for row in HEALTH_EXAMPLES:
            path = folder / row["file"]
            if path.resolve().parent != folder or digest(path) != row["sha256"]:
                raise ValueError("DINO 健全性測試圖片的路徑或雜湊不符。")
            with Image.open(path) as source:
                image = source.convert("RGB")
            if not any(overlap(box, row["xyxy"]) >= 0.5
                       for box, _ in detect(processor, model, device, image, row["prompt"])):
                raise RuntimeError(f"DINO 健全性測試未找到 {row['prompt']}：{path.name}")

    def ensure_loaded(self):
        if self.model is not None:
            return
        if self.load_error is not None:
            raise RuntimeError(self.load_error)
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            processor, model = load_model(self.directory, device)
            self._check_examples(processor, model, device)
        except Exception as exc:
            self.load_error = f"文字候選框已停用。請準備本機權重與對照圖後重新啟動。\n{exc}"
            raise RuntimeError(self.load_error) from exc
        self.processor, self.model, self.device = processor, model, device

    def propose(self, image, prompt, *, candidate_limit=DEFAULT_CANDIDATE_LIMIT):
        candidate_limit = validate_candidate_limit(candidate_limit)
        self.ensure_loaded()
        return detect(self.processor, self.model, self.device, image, prompt,
                      candidate_limit=candidate_limit)
