"""Vision adapter for the pinned llama.cpp b12541 CUDA ABI."""

from contextlib import ExitStack
import ctypes as c
from hashlib import sha256
import os
import sys
from time import perf_counter

import numpy as np

RUNTIME_VERSION = "1.0.0"
RUNTIME_COMMIT = "76f186df71a41abb3ee20fffedee31ca5dbc7d9f"
CONTEXT_TOKENS = 2048
DECODE_BATCH_TOKENS = 512

UINT, INT, SIZE, PTR = c.c_uint32, c.c_int32, c.c_size_t, c.c_void_p
TEXT, BOOL, FLOAT = c.c_char_p, c.c_bool, c.c_float


# By-value ABI from llama.cpp b12541 include/llama.h and tools/mtmd/mtmd.h.
class ModelParams(c.Structure):
    _fields_ = [
        ("devices", c.POINTER(PTR)), ("tensor_buft_overrides", PTR), ("n_gpu_layers", INT),
        ("split_mode", INT), ("load_mode", INT), ("lazy_mode", INT), ("main_gpu", INT),
        ("tensor_split", PTR), ("progress_callback", PTR), ("progress_callback_user_data", PTR),
        ("kv_overrides", PTR), ("vocab_only", BOOL), ("check_tensors", BOOL), ("use_extra_bufts", BOOL),
        ("no_host", BOOL), ("no_alloc", BOOL), ("load_mtp", BOOL),
    ]


class ContextParams(c.Structure):
    _fields_ = [
        ("n_ctx", UINT), ("n_batch", UINT), ("n_ubatch", UINT), ("n_seq_max", UINT), ("n_rs_seq", UINT),
        ("n_outputs_max", UINT), ("n_outputs_max_per_seq", UINT), ("n_threads", INT), ("n_threads_batch", INT),
        ("ctx_type", INT), ("rope_scaling_type", INT), ("pooling_type", INT), ("attention_type", INT),
        ("flash_attn_type", INT), ("rope_freq_base", FLOAT), ("rope_freq_scale", FLOAT),
        ("yarn_ext_factor", FLOAT), ("yarn_attn_factor", FLOAT), ("yarn_beta_fast", FLOAT),
        ("yarn_beta_slow", FLOAT), ("yarn_orig_ctx", UINT), ("cb_eval", PTR), ("cb_eval_user_data", PTR),
        ("type_k", INT), ("type_v", INT), ("moe_cache_size", SIZE), ("moe_cache_cpu_share", FLOAT),
        ("abort_callback", PTR), ("abort_callback_data", PTR), ("embeddings", BOOL), ("offload_kqv", BOOL),
        ("no_perf", BOOL), ("op_offload", BOOL), ("swa_full", BOOL), ("kv_unified", BOOL),
        ("samplers", PTR), ("n_samplers", SIZE), ("ctx_other", PTR),
    ]


class VisionParams(c.Structure):
    _fields_ = [
        ("use_gpu", BOOL), ("device", PTR), ("print_timings", BOOL), ("n_threads", INT),
        ("image_marker", TEXT), ("media_marker", TEXT), ("flash_attn_type", INT), ("warmup", BOOL),
        ("image_min_tokens", INT), ("image_max_tokens", INT), ("cb_eval", PTR), ("cb_eval_user_data", PTR),
        ("batch_max_tokens", INT), ("progress_callback", PTR), ("progress_callback_user_data", PTR),
    ]


class VisionText(c.Structure):
    _fields_ = [("text", TEXT), ("text_len", SIZE), ("add_special", BOOL), ("parse_special", BOOL)]


def function(library, name, result, *arguments):
    method = getattr(library, name)
    method.restype = result
    method.argtypes = arguments
    return method


def probabilities_from_logits(logits, token_ids, categories):
    choice_count = 4
    logits = np.asarray(logits, dtype=np.float64)
    if (logits.ndim != 1 or not np.isfinite(logits).all() or len(logits) < 100000
            or len(token_ids) != len(categories) or len(set(token_ids)) != len(token_ids)
            or set(categories) != set(range(choice_count)) or not all(0 <= token < len(logits) for token in token_ids)):
        raise ValueError("Full vocabulary logits and distinct answer token IDs are required.")
    weights = np.exp(logits - logits.max())
    allowed = weights[token_ids]
    total_allowed = allowed.sum()
    if total_allowed <= 0:
        raise ValueError("The answer token probability underflowed.")
    probabilities = np.bincount(categories, weights=allowed, minlength=choice_count) / total_allowed
    return probabilities.tolist(), float(total_allowed / weights.sum())


class NativeVision:
    def __init__(self, stack, runtime, model, mmproj, *, chat_prompt):
        if os.name != "nt" or c.sizeof(PTR) != 8:
            raise RuntimeError("The pinned runtime is for x64 Windows.")
        stack.enter_context(os.add_dll_directory(str(runtime)))
        self.llama = llama = c.CDLL(str(runtime / "llama.dll"))
        self.mtmd = mtmd = c.CDLL(str(runtime / "mtmd.dll"))
        backend = c.CDLL(str(runtime / "ggml.dll"))
        backend_base = c.CDLL(str(runtime / "ggml-base.dll"))
        self.version = function(llama, "llama_version", TEXT)().decode("ascii")
        self.commit = function(backend_base, "ggml_commit", TEXT)().decode("ascii")
        if (self.version not in (RUNTIME_VERSION, RUNTIME_VERSION + "-dev")
                or len(self.commit) < 7 or not RUNTIME_COMMIT.startswith(self.commit)):
            raise RuntimeError(f"llama.cpp ABI version mismatch: {self.version} ({self.commit})")
        # Explicit registration emits useful errors and avoids loading unrelated DLL backends.
        callback_type = c.CFUNCTYPE(None, INT, TEXT, PTR)

        def log_message(_level, message, _data):
            if message:
                sys.stderr.write(message.decode("utf-8", errors="replace"))
                sys.stderr.flush()

        self.log_callback = callback_type(log_message)
        set_log = function(backend_base, "ggml_log_set", None, callback_type, PTR)
        set_log(self.log_callback, None)
        stack.callback(set_log, callback_type(), None)
        self.backend_libraries = []
        load_backend = function(backend, "ggml_backend_load", PTR, TEXT)
        for name in ("ggml-cuda.dll", "ggml-cpu-x64.dll"):
            path = runtime / name
            # Preload dependencies with the safe ctypes search rooted at the verified folder.
            self.backend_libraries.append(c.CDLL(str(path)))
            if not load_backend(os.fsencode(path)):
                raise RuntimeError(f"ggml could not register {name}.")
        function(llama, "llama_backend_init", None)()
        stack.callback(function(llama, "llama_backend_free", None))
        self.device = function(backend, "ggml_backend_dev_by_name", PTR, TEXT)(b"CUDA0")
        if not self.device or not function(llama, "llama_supports_gpu_offload", BOOL)():
            device_count = function(backend, "ggml_backend_dev_count", SIZE)()
            device_at = function(backend, "ggml_backend_dev_get", PTR, SIZE)
            device_name = function(backend_base, "ggml_backend_dev_name", TEXT, PTR)
            devices = [device_name(device_at(index)).decode("utf-8") for index in range(device_count)]
            raise RuntimeError(f"The pinned runtime could not initialize CUDA0. Registered devices: {devices}")
        self.gpu_name = function(backend_base, "ggml_backend_dev_description", TEXT, PTR)(self.device).decode("utf-8")
        self.read_memory = function(backend_base, "ggml_backend_dev_memory", None, PTR, c.POINTER(SIZE), c.POINTER(SIZE))
        self.memory = [self.memory_snapshot("before_model_load")]
        self.chat_prompt = chat_prompt

        started = perf_counter()
        params = function(llama, "llama_model_default_params", ModelParams)()
        self.devices = (PTR * 2)(self.device, None)
        params.devices = self.devices
        params.split_mode = 0
        params.n_gpu_layers = -1
        self.model = function(llama, "llama_model_load_from_file", PTR, TEXT, ModelParams)(os.fsencode(model), params)
        if not self.model:
            raise RuntimeError("Could not load the GGUF text model on CUDA0.")
        stack.callback(function(llama, "llama_model_free", None, PTR), self.model)
        self.memory.append(self.memory_snapshot("after_model_load"))

        ctx = function(llama, "llama_context_default_params", ContextParams)()
        ctx.n_ctx, ctx.n_batch, ctx.n_ubatch, ctx.n_seq_max = CONTEXT_TOKENS, DECODE_BATCH_TOKENS, DECODE_BATCH_TOKENS, 1
        ctx.flash_attn_type = 0
        self.context = function(llama, "llama_init_from_model", PTR, PTR, ContextParams)(self.model, ctx)
        if not self.context:
            raise RuntimeError("Could not initialize the GGUF context on CUDA0.")
        stack.callback(function(llama, "llama_free", None, PTR), self.context)
        self.context_size = function(llama, "llama_n_ctx", UINT, PTR)(self.context)

        vision = function(mtmd, "mtmd_context_params_default", VisionParams)()
        vision.device = self.device
        vision.use_gpu = True
        vision.flash_attn_type = 0
        vision.warmup = False
        self.vision = function(mtmd, "mtmd_init_from_file", PTR, TEXT, PTR, VisionParams)(
            os.fsencode(mmproj), self.model, vision)
        if not self.vision:
            raise RuntimeError("Could not load the GGUF vision projector on CUDA0.")
        stack.callback(function(mtmd, "mtmd_free", None, PTR), self.vision)
        if not function(mtmd, "mtmd_support_vision", BOOL, PTR)(self.vision):
            raise RuntimeError("The supplied projector has no vision support.")
        self.marker = function(mtmd, "mtmd_default_marker", TEXT)()
        if not self.marker:
            raise RuntimeError("No multimodal marker was returned by libmtmd.")

        self.vocab = function(llama, "llama_model_get_vocab", PTR, PTR)(self.model)
        self.vocab_size = function(llama, "llama_vocab_n_tokens", INT, PTR)(self.vocab)
        self.tokenize = function(llama, "llama_tokenize", INT, PTR, TEXT, INT, c.POINTER(INT), INT, BOOL, BOOL)
        self.get_logits = function(llama, "llama_get_logits_ith", c.POINTER(FLOAT), PTR, INT)
        self.memory_clear = function(llama, "llama_memory_clear", None, PTR, BOOL)
        self.context_memory = function(llama, "llama_get_memory", PTR, PTR)(self.context)
        if not self.context_memory or not 100000 <= self.vocab_size <= 300000:
            raise RuntimeError("Unexpected GGUF context memory or vocabulary size.")
        self.load_seconds = perf_counter() - started
        self.memory.append(self.memory_snapshot("after_context_and_projector"))

    def memory_snapshot(self, stage):
        free, total = SIZE(), SIZE()
        self.read_memory(self.device, c.byref(free), c.byref(total))
        return {"stage": stage, "total_bytes": total.value, "free_bytes": free.value,
                "device_used_bytes": total.value - free.value}

    def encode(self, text, add_special_tokens=False):
        data = text.encode("utf-8")
        tokens = (INT * 8)()
        count = self.tokenize(self.vocab, data, len(data), tokens, len(tokens), add_special_tokens, False)
        if not 0 < count <= len(tokens):
            raise RuntimeError("Unable to tokenize the model's answer letter.")
        return list(tokens[:count])

    def vote(self, image, text):
        mtmd = self.mtmd
        started = perf_counter()
        prompt = self.chat_prompt(text, self.marker)
        rgb = image.tobytes()
        self.memory_clear(self.context_memory, True)
        with ExitStack() as stack:
            bitmap = function(mtmd, "mtmd_bitmap_init", PTR, UINT, UINT, TEXT)(image.width, image.height, rgb)
            if not bitmap:
                raise RuntimeError("Could not create the RGB vision input.")
            stack.callback(function(mtmd, "mtmd_bitmap_free", None, PTR), bitmap)
            chunks = function(mtmd, "mtmd_input_chunks_init", PTR)()
            if not chunks:
                raise RuntimeError("Could not allocate the multimodal token chunks.")
            stack.callback(function(mtmd, "mtmd_input_chunks_free", None, PTR), chunks)
            image_inputs = (PTR * 1)(bitmap)
            tokenize = function(mtmd, "mtmd_tokenize", INT, PTR, PTR, c.POINTER(VisionText), c.POINTER(PTR), SIZE)
            if tokenize(self.vision, chunks, c.byref(VisionText(prompt, len(prompt), True, True)), image_inputs, 1):
                raise RuntimeError("Could not combine the prompt with the one RGB image.")
            chunk_count = function(mtmd, "mtmd_input_chunks_size", SIZE, PTR)(chunks)
            get_chunk = function(mtmd, "mtmd_input_chunks_get", PTR, PTR, SIZE)
            get_type = function(mtmd, "mtmd_input_chunk_get_type", INT, PTR)
            chunk_types = [get_type(get_chunk(chunks, index)) for index in range(chunk_count)]
            token_count = function(mtmd, "mtmd_helper_get_n_tokens", SIZE, PTR)(chunks)
            if chunk_types.count(1) != 1 or not chunk_types or chunk_types[-1] != 0 or token_count > self.context_size:
                raise RuntimeError("Expected one image chunk followed by text within the context capacity.")
            end = INT()
            evaluate = function(mtmd, "mtmd_helper_eval_chunks", INT, PTR, PTR, PTR, INT, INT, INT, BOOL, c.POINTER(INT))
            if evaluate(self.vision, self.context, chunks, 0, 0, DECODE_BATCH_TOKENS, True, c.byref(end)):
                raise RuntimeError("Multimodal prompt evaluation failed before producing logits.")
            pointer = self.get_logits(self.context, -1)
            if not pointer:
                raise RuntimeError("No last-token logits were returned by llama.cpp.")
            logits = np.ctypeslib.as_array(pointer, shape=(self.vocab_size,)).copy()
        return logits, {"seconds": round(perf_counter() - started, 6), "input_tokens": token_count,
                        "image_chunks": chunk_types.count(1), "sequence_end_position": end.value,
                        "prompt_sha256": sha256(prompt).hexdigest(),
                        "logits_sha256": sha256(logits.astype("<f4", copy=False).tobytes()).hexdigest()}
