"""GGUF container reader, ggml codec dequantization, and qwen35/qwen35moe mapping.

Grounding (all verified against the vendored llama.cpp checkout, not memory):

- Container: ``ggml/include/gguf.h`` — magic ``"GGUF"``, version 3, tensor/KV
  counts, typed KV, per-tensor name/dim-count/``ne``/type/offset, then the
  aligned data blob. Offsets are relative to the blob start.
- Codec semantics: ``ggml/src/ggml-quants.c`` ``dequantize_row_*`` and the
  block structs in ``ggml/src/ggml-common.h`` (Q8_0 32/34B, Q4_K 256/144B,
  Q5_K 256/176B, Q6_K 256/210B, NVFP4 64/36B with 4 UE4M3 scales first).
- qwen35/qwen35moe tensor names: gguf-py ``TensorNameMap`` canonical names
  (``token_embd``, ``blk.{bid}.attn_q``, ``ffn_gate_exps``, ...).
- NVFP4 sidecars: ``conversion/base.py`` ``_repack_nvfp4`` writes
  ``<name>.scale`` and ``<name>.input_scale`` holding the inference-time
  *multipliers* — i.e. the reciprocals of the HF global scales — so the
  ``EncodedRows`` divisors here are the reciprocals of the sidecar values.
  NVFP4 code packing differs from NInfer's (ggml pairs elements ``(j, j+8)``
  per sub-block per ``dequantize_row_nvfp4``; NInfer pairs ``(2j, 2j+1)``),
  so encoded import reframes the nibbles, it does not copy words.
- GDN V-head order: ``conversion/qwen.py`` ``_LinearAttentionVReorderBase``
  permutes V heads from grouped to tiled order when writing a GGUF
  (``num_k_heads != num_v_heads``); the permutation is an involution, so
  reading applies the same function to restore HF order.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct

import torch

from tools.artifact.file_io import discard_cached_pages
from .logical import EncodedRows, LogicalSource, select_rows, transpose_source

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_DEFAULT_ALIGNMENT = 32

# ggml_type ids (ggml/include/ggml.h). Only the types a Qwen-family GGUF
# plausibly contains are implemented; anything else is a clear error.
_F32, _F16, _Q8_0, _Q4_K, _Q5_K, _Q6_K, _BF16, _NVFP4 = 0, 1, 8, 12, 13, 14, 30, 40


@dataclass(frozen=True, slots=True)
class _GgmlType:
    name: str
    block: int
    block_bytes: int


_GGML_TYPES = {
    _F32: _GgmlType("F32", 1, 4),
    _F16: _GgmlType("F16", 1, 2),
    _Q8_0: _GgmlType("Q8_0", 32, 34),
    _Q4_K: _GgmlType("Q4_K", 256, 144),
    _Q5_K: _GgmlType("Q5_K", 256, 176),
    _Q6_K: _GgmlType("Q6_K", 256, 210),
    _BF16: _GgmlType("BF16", 1, 2),
    _NVFP4: _GgmlType("NVFP4", 64, 36),
}

# gguf_type ids (ggml/include/gguf.h).
_U8, _I8, _U16, _I16, _U32, _I32, _F32_T, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(
    13
)
_SCALAR_FORMATS = {
    _U8: ("<B", 1),
    _I8: ("<b", 1),
    _U16: ("<H", 2),
    _I16: ("<h", 2),
    _U32: ("<I", 4),
    _I32: ("<i", 4),
    _F32_T: ("<f", 4),
    _U64: ("<Q", 8),
    _I64: ("<q", 8),
    _F64: ("<d", 8),
}

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
# Q6_K lanes per 128-element chunk: (ql offset, nibble shift, qh shift, scale slot),
# matching dequantize_row_q6_K (q1..q4 order); lane k fills y[k*32:(k+1)*32].
_Q6K_SLOTS = ((0, 0, 0, 0), (32, 0, 2, 2), (0, 4, 4, 4), (32, 4, 6, 6))


def _words(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype).float()


def _fp16(data: bytes) -> torch.Tensor:
    return _words(data, torch.float16)


def _dequant_q8_0(raw: bytes) -> torch.Tensor:
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 34)
    scales = _fp16(blocks[:, 0:2].numpy().tobytes())[:, None]
    return (blocks[:, 2:].to(torch.int8).float() * scales).reshape(-1)


def _k_scales_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the 12 K-quant scale bytes to 8 (scale, min) pairs (get_scale_min_k4)."""
    sc = torch.empty((*scales.shape[:-1], 8), dtype=torch.int64)
    mn = torch.empty((*scales.shape[:-1], 8), dtype=torch.int64)
    sc[..., :4] = scales[..., :4] & 63
    mn[..., :4] = scales[..., 4:8] & 63
    j = torch.arange(4, 8)
    sc[..., 4:] = (scales[..., j + 4] & 15) | ((scales[..., j - 4] >> 6) << 4)
    mn[..., 4:] = (scales[..., j + 4] >> 4) | ((scales[..., j] >> 6) << 4)
    return sc, mn


def _dequant_q4_k(raw: bytes) -> torch.Tensor:
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 144)
    d = _fp16(blocks[:, 0:2].numpy().tobytes())
    dmin = _fp16(blocks[:, 2:4].numpy().tobytes())
    sc, mn = _k_scales_min(blocks[:, 4:16])
    nibbles = blocks[:, 16:].reshape(-1, 4, 32)
    low, high = nibbles & 15, nibbles >> 4
    first = d[:, None, None] * sc[:, ::2, None] * low - dmin[:, None, None] * mn[
        :, ::2, None
    ]
    second = d[:, None, None] * sc[:, 1::2, None] * high - dmin[:, None, None] * mn[
        :, 1::2, None
    ]
    return torch.stack((first, second), dim=2).reshape(-1)


def _dequant_q5_k(raw: bytes) -> torch.Tensor:
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 176)
    d = _fp16(blocks[:, 0:2].numpy().tobytes())
    dmin = _fp16(blocks[:, 2:4].numpy().tobytes())
    sc, mn = _k_scales_min(blocks[:, 4:16])
    # qh is shared across the four 64-groups: bits (2g, 2g+1) feed group g's low/high.
    high_bits = blocks[:, 16:48]
    nibbles = blocks[:, 48:].reshape(-1, 4, 32)
    group = torch.arange(4).reshape(1, 4, 1)
    low = (nibbles & 15) + (((high_bits[:, None, :] >> (2 * group)) & 1) * 16)
    high = (nibbles >> 4) + (((high_bits[:, None, :] >> (2 * group + 1)) & 1) * 16)
    first = d[:, None, None] * sc[:, ::2, None] * low - dmin[:, None, None] * mn[
        :, ::2, None
    ]
    second = d[:, None, None] * sc[:, 1::2, None] * high - dmin[:, None, None] * mn[
        :, 1::2, None
    ]
    return torch.stack((first, second), dim=2).reshape(-1)


def _dequant_q6_k(raw: bytes) -> torch.Tensor:
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 210)
    nb = blocks.shape[0]
    ql = blocks[:, :128].reshape(nb, 2, 64)
    qh = blocks[:, 128:192].reshape(nb, 2, 32)
    sc = blocks[:, 192:208].to(torch.int8).float().reshape(nb, 2, 8)
    d = _fp16(blocks[:, 208:210].numpy().tobytes()).reshape(nb, 1, 1)
    sub = torch.arange(32) // 16
    lanes = []
    for index, (offset, shift, bit, slot) in enumerate(_Q6K_SLOTS):
        nibble = (ql[:, :, offset : offset + 32] >> shift) & 15
        signed = ((nibble | (((qh >> bit) & 3) << 4)).to(torch.int8) - 32).float()
        scale = torch.stack(
            [half[:, sub + slot] for half in (sc[:, 0], sc[:, 1])], dim=1
        )
        lanes.append(d * scale * signed)
    return torch.cat(lanes, dim=2).reshape(-1)


def _ue4m3_to_fp32(codes: torch.Tensor) -> torch.Tensor:
    flat = (codes & 0x7F).to(torch.uint8).contiguous().reshape(-1)
    return flat.view(torch.float8_e4m3fn).float().reshape(codes.shape)


def _dequant_nvfp4(raw: bytes) -> torch.Tensor:
    """Decode NVFP4 super-blocks to FP32 (dequantize_row_nvfp4 semantics)."""
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 36)
    scales = _ue4m3_to_fp32(blocks[:, :4].reshape(-1, 4, 1))
    packed = blocks[:, 4:].reshape(-1, 4, 8)
    low = _E2M1[(packed & 15).long()]
    high = _E2M1[(packed >> 4).long()]
    grouped = torch.stack((low, high), dim=2).reshape(-1, 64)
    return (grouped * scales.repeat(1, 1, 16).reshape(-1, 64)).reshape(-1)


def _reframe_nvfp4(raw: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """Reframe ggml NVFP4 super-blocks into NInfer code/scale words.

    ggml pairs elements (j, j+8) per 16-sub-block; NInfer pairs (2j, 2j+1).
    Scales pass through (UE4M3 bits equal non-negative E4M3 bits).
    """
    blocks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, 36)
    scales = blocks[:, :4].reshape(-1, 64 // 16) & 0x7F
    if bool((scales > 0x7E).any()):
        raise ValueError("NVFP4 scales must be finite non-negative E4M3")
    packed = blocks[:, 4:].reshape(-1, 4, 8)
    low, high = packed & 15, packed >> 4
    elements = torch.stack((low, high), dim=2).reshape(-1, 64)
    pairs = elements.reshape(-1, 32, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).reshape(-1, 32), scales.reshape(-1, 4)
@dataclass(frozen=True, slots=True)
class GgufTensorInfo:
    name: str
    shape: tuple[int, ...]  # logical C-order shape (fastest axis last)
    ggml: str
    offset: int  # absolute file offset of the tensor payload
    bytes: int


class _Cursor:
    """Bounded sequential reads over one open binary stream."""

    def __init__(self, stream, size: int, label: str) -> None:
        self._stream = stream
        self._size = size
        self._label = label
        self.position = 0

    def read(self, count: int) -> bytes:
        if count < 0 or self.position + count > self._size:
            raise ValueError(f"{self._label}: truncated GGUF header")
        data = self._stream.read(count)
        if len(data) != count:
            raise ValueError(f"{self._label}: truncated GGUF header")
        self.position += count
        return data

    def skip(self, count: int) -> None:
        if count < 0 or self.position + count > self._size:
            raise ValueError(f"{self._label}: truncated GGUF header")
        self._stream.seek(count, os.SEEK_CUR)
        self.position += count


def _read_string(cursor: _Cursor) -> bytes:
    (length,) = struct.unpack("<Q", cursor.read(8))
    return cursor.read(length)


def _read_scalar(cursor: _Cursor, kind: int):
    if kind == _STR:
        return _read_string(cursor).decode("utf-8")
    if kind == _BOOL:
        return bool(struct.unpack("<b", cursor.read(1))[0])
    try:
        fmt, _ = _SCALAR_FORMATS[kind]
    except KeyError as error:
        raise ValueError(f"unsupported GGUF metadata type {kind}") from error
    return struct.unpack(fmt, cursor.read(struct.calcsize(fmt)))[0]


def _read_metadata(cursor: _Cursor, kind: int, *, store_strings: bool):
    if kind == _ARR:
        (item,) = struct.unpack("<i", cursor.read(4))
        (count,) = struct.unpack("<Q", cursor.read(8))
        if item == _STR and not store_strings:
            for _ in range(count):
                _read_string(cursor)
            return count
        return [_read_metadata(cursor, item, store_strings=True) for _ in range(count)]
    return _read_scalar(cursor, kind)


class GgufSource:
    """Bounded reads over one GGUF file; retains a single file descriptor."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        size = os.path.getsize(self.path)
        with self.path.open("rb") as stream:
            cursor = _Cursor(stream, size, str(self.path))
            if cursor.read(4) != _GGUF_MAGIC:
                raise ValueError(f"{self.path}: not a GGUF file")
            (version,) = struct.unpack("<I", cursor.read(4))
            if version != _GGUF_VERSION:
                raise ValueError(f"{self.path}: unsupported GGUF version {version}")
            (tensors,) = struct.unpack("<q", cursor.read(8))
            (pairs,) = struct.unpack("<q", cursor.read(8))
            if not 0 <= tensors <= size // 32 or not 0 <= pairs <= size // 8:
                raise ValueError(f"{self.path}: implausible GGUF counts")
            kv: dict[str, object] = {}
            for _ in range(pairs):
                key = _read_string(cursor).decode("utf-8")
                (kind,) = struct.unpack("<i", cursor.read(4))
                keep = key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
                kv[key] = _read_metadata(cursor, kind, store_strings=not keep)
            self._metadata = kv
            raw_infos = []
            for _ in range(tensors):
                name = _read_string(cursor).decode("utf-8")
                (dims,) = struct.unpack("<I", cursor.read(4))
                if not 1 <= dims <= 4:
                    raise ValueError(f"{name}: unsupported GGUF rank {dims}")
                ne = struct.unpack("<" + "q" * dims, cursor.read(8 * dims))
                if any(axis <= 0 for axis in ne):
                    raise ValueError(f"{name}: invalid GGUF dimensions {ne}")
                (type_id,) = struct.unpack("<i", cursor.read(4))
                try:
                    ggml = _GGML_TYPES[type_id]
                except KeyError as error:
                    raise ValueError(
                        f"{name}: unsupported GGUF type id {type_id}"
                    ) from error
                (relative,) = struct.unpack("<Q", cursor.read(8))
                raw_infos.append((name, ne, ggml, relative))
            alignment = kv.get("general.alignment", _GGUF_DEFAULT_ALIGNMENT)
            if type(alignment) is not int or alignment <= 0:
                raise ValueError(f"{self.path}: invalid general.alignment {alignment!r}")
            blob = cursor.position + (-cursor.position % alignment)
            infos = {}
            for name, ne, ggml, relative in raw_infos:
                shape = tuple(reversed(ne))
                if len(shape) > 1 and shape[-1] % ggml.block:
                    raise ValueError(
                        f"{name}: {ggml.name} row length {shape[-1]} "
                        f"is not a multiple of {ggml.block}"
                    )
                count = 1
                for axis in shape:
                    count *= axis
                nbytes = count // ggml.block * ggml.block_bytes
                offset = blob + relative
                if offset + nbytes > size:
                    raise ValueError(f"{name}: GGUF tensor range exceeds file")
                if name in infos:
                    raise ValueError(f"{self.path}: duplicate GGUF tensor {name!r}")
                infos[name] = GgufTensorInfo(name, shape, ggml.name, offset, nbytes)
            self._tensors = infos
        self.root = self.path.parent
        self.bytes_read = 0
        self._fd: int | None = None
        self._config: dict | None = None
        self._ordered: dict = {}

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = _synth_config(self._metadata, self)
        return self._config

    def _file(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def has(self, name: str) -> bool:
        return name in self._tensors

    def describe(self, name: str) -> GgufTensorInfo:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise ValueError(f"{self.path}: missing source tensor {name!r}") from error

    def metadata(self, key: str):
        return self._metadata.get(key)

    def token_count(self) -> int:
        count = self._metadata.get("tokenizer.ggml.tokens")
        if type(count) is not int:
            raise ValueError(f"{self.path}: GGUF vocabulary size is unavailable")
        return count

    def _payload(self, info: GgufTensorInfo, begin: int, end: int) -> bytes:
        fd = self._file()
        raw = os.pread(fd, end - begin, info.offset + begin)
        if len(raw) != end - begin:
            raise ValueError(f"{info.name}: short source read")
        self.bytes_read += end - begin
        discard_cached_pages(fd, info.offset + begin, end - begin)
        return raw

    def close(self) -> None:
        if self._fd is not None:
            discard_cached_pages(self._fd)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> GgufSource:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


_QWEN35_ARCHES = {"qwen35": False, "qwen35moe": True}


def _synth_config(kv: dict[str, object], store: GgufSource) -> dict:
    """Synthesize the HF-style config text_config consumes, from GGUF KV metadata."""
    arch = kv.get("general.architecture")
    if arch not in _QWEN35_ARCHES:
        raise ValueError(f"{store.path}: unsupported GGUF architecture {arch!r}")
    moe = _QWEN35_ARCHES[arch]
    prefix = arch

    def field(key: str, kind: str):
        value = kv.get(f"{prefix}.{key}")
        if kind == "int" and type(value) is not int:
            raise ValueError(f"{store.path}: GGUF metadata {prefix}.{key} is missing")
        if kind == "float" and type(value) not in (float, int):
            raise ValueError(f"{store.path}: GGUF metadata {prefix}.{key} is missing")
        return value

    config = {
        "architectures": [
            "Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"
        ],
        "hidden_size": field("embedding_length", "int"),
        "vocab_size": store.token_count(),
        "num_hidden_layers": field("block_count", "int"),
        "max_position_embeddings": field("context_length", "int"),
        "tie_word_embeddings": not store.has("output.weight"),
        "rms_norm_eps": kv.get(f"{prefix}.attention.layer_norm_rms_epsilon", 1e-6),
        "num_attention_heads": field("attention.head_count", "int"),
        "num_key_value_heads": field("attention.head_count_kv", "int"),
    }
    key_length = field("attention.key_length", "int")
    if field("attention.value_length", "int") != key_length:
        raise ValueError(f"{store.path}: mismatched GGUF key/value lengths")
    config["head_dim"] = key_length
    recurrent = kv.get(f"{prefix}.attention.recurrent_layers")
    if recurrent is not None:
        if (
            not isinstance(recurrent, list)
            or len(recurrent) != config["num_hidden_layers"]
            or any(type(item) is not bool for item in recurrent)
        ):
            raise ValueError(f"{store.path}: invalid recurrent_layers metadata")
        config["layer_types"] = [
            "linear_attention" if item else "full_attention" for item in recurrent
        ]
    else:
        interval = kv.get(f"{prefix}.full_attention_interval", 4)
        if type(interval) is not int or interval <= 0:
            raise ValueError(f"{store.path}: invalid full_attention_interval")
        config["layer_types"] = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(config["num_hidden_layers"])
        ]
    sections = kv.get(f"{prefix}.rope.dimension_sections")
    if (
        not isinstance(sections, list)
        or len(sections) < 3
        or any(type(item) is not int for item in sections[:3])
    ):
        raise ValueError(f"{store.path}: GGUF RoPE dimension sections are missing")
    dimension_count = kv.get(f"{prefix}.rope.dimension_count", 0)
    config["rope_parameters"] = {
        "mrope_section": list(sections[:3]),
        "rope_theta": kv.get(f"{prefix}.rope.freq_base", 10_000_000),
        "partial_rotary_factor": (
            dimension_count / key_length if dimension_count else 0.25
        ),
    }
    if "linear_attention" in config["layer_types"]:
        keys = field("ssm.group_count", "int")
        config["linear_num_key_heads"] = keys
        config["linear_key_head_dim"] = field("ssm.state_size", "int")
        config["linear_num_value_heads"] = field("ssm.time_step_rank", "int")
        inner = field("ssm.inner_size", "int")
        if inner % config["linear_num_value_heads"]:
            raise ValueError(f"{store.path}: GGUF SSM inner size is inconsistent")
        config["linear_value_head_dim"] = inner // config["linear_num_value_heads"]
        config["linear_conv_kernel_dim"] = field("ssm.conv_kernel", "int")
    if moe:
        config["num_experts"] = field("expert_count", "int")
        used = field("expert_used_count", "int")
        config["num_experts_per_tok"] = used
        config["moe_intermediate_size"] = field("expert_feed_forward_length", "int")
        config["shared_expert_intermediate_size"] = field(
            "expert_shared_feed_forward_length", "int"
        )
    else:
        config["intermediate_size"] = field("feed_forward_length", "int")
    embedding = store.describe("token_embd.weight")
    if embedding.shape != (config["vocab_size"], config["hidden_size"]):
        raise ValueError(
            f"{store.path}: token embedding shape {embedding.shape} "
            "disagrees with metadata"
        )
    return config


_TEXT_PREFIXES = ("model.language_model.", "model.")


@dataclass(frozen=True, slots=True)
class GgufRef:
    """A translated GGUF tensor reference: canonical base plus expert selection."""

    base: str
    expert: int | None = None


def _block_name(hf_name: str) -> tuple[str, str] | None:
    """Split an HF checkpoint name into (prefix, remainder)."""
    for prefix in _TEXT_PREFIXES:
        if hf_name.startswith(prefix):
            return prefix, hf_name[len(prefix) :]
    if hf_name == "lm_head.weight":
        return "", hf_name
    return None


def gguf_translate(hf_name: str) -> GgufRef | None:
    """Map an HF checkpoint tensor name to its canonical GGUF tensor.

    Returns None for names outside the qwen35/qwen35moe text mapping (vision,
    draft, and unknown names are not mapped).
    """
    split = _block_name(hf_name)
    if split is None:
        return None
    _, rest = split
    if rest == "embed_tokens.weight":
        return GgufRef("token_embd.weight")
    if rest == "lm_head.weight":
        return GgufRef("output.weight")
    if rest == "norm.weight":
        return GgufRef("output_norm.weight")
    if not rest.startswith("layers."):
        return None
    block, _, field = rest.partition(".")[2].partition(".")
    try:
        bid = int(block)
    except ValueError:
        return None
    base = f"blk.{bid}"
    if field == "input_layernorm.weight":
        return GgufRef(f"{base}.attn_norm.weight")
    if field == "post_attention_layernorm.weight":
        return GgufRef(f"{base}.post_attention_norm.weight")
    for role, canonical in (
        ("q_proj.weight", "attn_q.weight"),
        ("k_proj.weight", "attn_k.weight"),
        ("v_proj.weight", "attn_v.weight"),
        ("o_proj.weight", "attn_output.weight"),
        ("q_norm.weight", "attn_q_norm.weight"),
        ("k_norm.weight", "attn_k_norm.weight"),
    ):
        if field == f"self_attn.{role}":
            return GgufRef(f"{base}.{canonical}")
    for role in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        if field == f"mlp.{role}":
            return GgufRef(f"{base}.ffn_{role.removesuffix('_proj.weight')}.weight")
    if field == "mlp.gate.weight":
        return GgufRef(f"{base}.ffn_gate_inp.weight")
    if field == "mlp.shared_expert_gate.weight":
        return GgufRef(f"{base}.ffn_gate_inp_shexp.weight")
    if field.startswith("mlp.experts."):
        parts = field.split(".")
        if (
            len(parts) == 5
            and parts[4] == "weight"
            and parts[3] in ("gate_proj", "up_proj", "down_proj")
        ):
            try:
                expert = int(parts[2])
            except ValueError:
                return None
            role = parts[3].removesuffix("_proj")
            return GgufRef(f"{base}.ffn_{role}_exps.weight", expert=expert)
        if field in ("mlp.experts.gate_up_proj.weight", "mlp.experts.gate_up_proj"):
            return GgufRef(f"{base}.ffn_gate_up_exps.weight")
        if field in ("mlp.experts.down_proj.weight", "mlp.experts.down_proj"):
            return GgufRef(f"{base}.ffn_down_exps.weight")
        return None
    for role in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        if field == f"mlp.shared_expert.{role}":
            short = role.removesuffix("_proj.weight")
            return GgufRef(f"{base}.ffn_{short}_shexp.weight")
    gdn = {
        "in_proj_qkv.weight": ("attn_qkv.weight", "qkv"),
        "in_proj_z.weight": ("attn_gate.weight", "z"),
        "in_proj_a.weight": ("ssm_alpha.weight", "a"),
        "in_proj_b.weight": ("ssm_beta.weight", "b"),
        "conv1d.weight": ("ssm_conv1d.weight", "conv"),
        "A_log": ("ssm_a", "alog"),
        "dt_bias": ("ssm_dt.bias", "dt"),
        "norm.weight": ("ssm_norm.weight", None),
        "out_proj.weight": ("ssm_out.weight", "out"),
    }
    if field.startswith("linear_attn.") and field[len("linear_attn.") :] in gdn:
        canonical, _ = gdn[field[len("linear_attn.") :]]
        return GgufRef(f"{base}.{canonical}")
    return None


def _v_permutation(count: int, heads_k: int, per_k: int, head_dim: int) -> torch.Tensor:
    """GGUF write-side V-head tiling (grouped to tiled).

    Mirrors ``_LinearAttentionVReorderBase._reorder_v_heads``: the flat input
    is viewed as ``[heads_k, per_k, head_dim]`` and axes 0 and 1 are swapped.
    """
    grid = torch.arange(count).reshape(heads_k, per_k, head_dim)
    return grid.permute(1, 0, 2).reshape(-1)


def _v_inverse_permutation(
    count: int, heads_k: int, per_k: int, head_dim: int
) -> torch.Tensor:
    """Inverse tiling (tiled back to grouped HF order).

    An axis swap is its own inverse only for equal axes; in general the
    inverse views the tiled layout as ``[per_k, heads_k, head_dim]``. The two
    coincide when ``heads_k == per_k``.
    """
    return _v_permutation(count, per_k, heads_k, head_dim)


def _restore_order(
    values: torch.Tensor,
    role: str | None,
    heads_k: int,
    keys_k: int,
    values_k: int,
    head_v: int,
) -> torch.Tensor:
    """Undo the GGUF V-head tiling for one GDN tensor (values already decoded)."""
    if role is None or heads_k == values_k:
        return values
    per_k = values_k // heads_k
    keep = 2 * heads_k * keys_k
    if role == "out":
        perm = _v_inverse_permutation(values.shape[1], heads_k, per_k, head_v)
        return values[:, perm]
    if role == "conv":
        channels = values.shape[0]
        tiled = values_k * head_v
        perm = torch.cat(
            (
                torch.arange(keep),
                keep + _v_inverse_permutation(tiled, heads_k, per_k, head_v),
            )
        )
        if perm.numel() != channels:
            raise ValueError("GGUF conv1d channel count disagrees with SSM geometry")
        return values[perm]
    if role == "qkv":
        perm = torch.cat(
            (
                torch.arange(keep),
                keep + _v_inverse_permutation(values.shape[0] - keep, heads_k, per_k, head_v),
            )
        )
        if perm.numel() != values.shape[0]:
            raise ValueError("GGUF qkv row count disagrees with SSM geometry")
        return values[perm]
    head = 1 if role in ("a", "b", "alog", "dt") else head_v
    perm = _v_inverse_permutation(values.shape[0], heads_k, per_k, head)
    if perm.numel() != values.shape[0]:
        raise ValueError("GGUF tensor length disagrees with SSM geometry")
    return values[perm]


def _restore_encoded(
    codes: torch.Tensor,
    scales: torch.Tensor,
    role: str | None,
    heads_k: int,
    keys_k: int,
    values_k: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo the GGUF V-head tiling on NVFP4 code/scale words (pre-quant reorder)."""
    if role is None or heads_k == values_k:
        return codes, scales
    per_k = values_k // heads_k
    if role == "out":
        cols = codes.shape[1] * 2
        forward = _v_permutation(cols, heads_k, per_k, head_v)
        inverse = torch.empty_like(forward)
        inverse[forward] = torch.arange(cols)
        groups = torch.arange(cols) // 16
        mapped = groups[inverse] // 16
        if (mapped.reshape(-1, 16) != mapped.reshape(-1, 16)[:, :1]).any():
            raise ValueError("GGUF out_proj column reorder breaks NVFP4 groups")
        source_group = mapped[::16]
        if not torch.equal(
            torch.sort(source_group).values,
            torch.arange(cols // 16),
        ):
            raise ValueError("GGUF out_proj column reorder breaks NVFP4 groups")
        elements = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(
            codes.shape[0], cols
        )[:, inverse]
        pairs = elements.reshape(codes.shape[0], -1, 2)
        codes = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
        return codes, scales[:, source_group]
    rows = codes.shape[0]
    keep = 2 * heads_k * keys_k
    if role == "qkv":
        perm = torch.cat(
            (torch.arange(keep), keep + _v_inverse_permutation(rows - keep, heads_k, per_k, head_v))
        )
    elif role == "conv":
        perm = torch.cat(
            (torch.arange(keep), keep + _v_inverse_permutation(rows - keep, heads_k, per_k, head_v))
        )
    else:
        head = 1 if role in ("a", "b") else head_v
        perm = _v_inverse_permutation(rows, heads_k, per_k, head)
    if perm.numel() != rows:
        raise ValueError("GGUF tensor length disagrees with SSM geometry")
    return codes[perm], scales[perm]


def _geometry(store: GgufSource) -> tuple[int, int, int, int]:
    try:
        return (
            store.config["linear_num_key_heads"],
            store.config["linear_key_head_dim"],
            store.config["linear_num_value_heads"],
            store.config["linear_value_head_dim"],
        )
    except KeyError as error:
        raise ValueError(
            f"{store.path}: GGUF GDN tensor without SSM geometry"
        ) from error
_GDN_ROLES = {
    "attn_qkv.weight": "qkv",
    "attn_gate.weight": "z",
    "ssm_alpha.weight": "a",
    "ssm_beta.weight": "b",
    "ssm_conv1d.weight": "conv",
    "ssm_a": "alog",
    "ssm_dt.bias": "dt",
    "ssm_out.weight": "out",
}

_PLAIN = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
_DECODERS = {
    "Q8_0": _dequant_q8_0,
    "Q4_K": _dequant_q4_k,
    "Q5_K": _dequant_q5_k,
    "Q6_K": _dequant_q6_k,
    "NVFP4": _dequant_nvfp4,
}


def _undo_write_transform(values: torch.Tensor, hf_name: str) -> torch.Tensor:
    """Invert llama.cpp's write-side value transforms (conversion/qwen.py).

    ``Qwen3NextModel.modify_tensors`` stores ``A_log`` as ``-exp(A_log)`` and
    adds one to every ``norm.weight`` except the GDN norm; ``dt_bias`` is only
    renamed. All three are pointwise, so they commute with the V-head tiling.
    """
    if hf_name.endswith(".A_log"):
        if bool((values >= 0).any()):
            raise ValueError(f"{hf_name}: GGUF A_log must be negative")
        return torch.log(-values)
    if hf_name.endswith("norm.weight") and ".linear_attn.norm.weight" not in hf_name:
        return values - 1
    return values


def _role_of(base: str) -> str | None:
    parts = base.split(".", 2)
    return _GDN_ROLES.get(parts[2] if len(parts) == 3 else base)


def _values_flat(
    store: GgufSource, info: GgufTensorInfo, begin: int, end: int
) -> torch.Tensor:
    """Decode flat elements [begin, end) to FP32 (bounded block-cover reads)."""
    count = 1
    for axis in info.shape:
        count *= axis
    if not 0 <= begin <= end <= count:
        raise ValueError(f"{info.name}: element range [{begin},{end}) exceeds {info.shape}")
    if begin == end:
        return torch.empty(0, dtype=torch.float32)
    if info.ggml in _PLAIN:
        width = _GGML_TYPES[
            next(k for k, v in _GGML_TYPES.items() if v.name == info.ggml)
        ].block_bytes
        raw = store._payload(info, begin * width, end * width)
        return _words(raw, _PLAIN[info.ggml])
    ggml = next(v for v in _GGML_TYPES.values() if v.name == info.ggml)
    first, last = begin // ggml.block, (end + ggml.block - 1) // ggml.block
    raw = store._payload(
        info, first * ggml.block_bytes, last * ggml.block_bytes
    )
    decoded = _DECODERS[info.ggml](raw)
    return decoded[begin - first * ggml.block : end - first * ggml.block]


def _ordered_values(store: GgufSource, base: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Full decoded tensor with the GGUF V-head tiling undone (memoized)."""
    key = ("values", base)
    cached = store._ordered.get(key)
    if cached is None:
        info = store.describe(base)
        flat = _values_flat(store, info, 0, _count(info.shape))
        values = flat.reshape(info.shape)
        role = _role_of(base)
        if role is not None:
            heads_k, keys_k, values_k, head_v = _geometry(store)
            values = _restore_order(values, role, heads_k, keys_k, values_k, head_v)
        if tuple(values.shape) != tuple(shape) and _count(values.shape) == _count(shape):
            values = values.reshape(shape)
        cached = store._ordered.setdefault(key, values)
    return cached


def _ordered_codes(
    store: GgufSource, base: str, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full NVFP4 code/scale words with the V-head tiling undone (memoized)."""
    key = ("codes", base)
    cached = store._ordered.get(key)
    if cached is None:
        rows, cols = shape
        codes, scales = _encoded_span(store, base, 0, rows)
        role = _role_of(base)
        if role is not None:
            heads_k, keys_k, values_k, head_v = _geometry(store)
            codes, scales = _restore_encoded(
                codes, scales, role, heads_k, keys_k, values_k, head_v
            )
        cached = store._ordered.setdefault(key, (codes, scales))
    return cached


def _matrix_values(
    store: GgufSource, base: str, shape: tuple[int, int], begin: int, end: int
) -> torch.Tensor:
    info = store.describe(base)
    if tuple(info.shape) != shape:
        raise ValueError(f"{base}: expected matrix shape {shape}, got {info.shape}")
    if _role_of(base) is None:
        return _values_flat(store, info, begin, end)
    return _ordered_values(store, base, shape).reshape(-1)[begin:end]


def _encoded_span(
    store: GgufSource, base: str, first: int, last: int
) -> tuple[torch.Tensor, torch.Tensor]:
    info = store.describe(base)
    if info.ggml != "NVFP4" or len(info.shape) != 2:
        raise ValueError(f"{base}: NVFP4 encoded rows require a 2-D NVFP4 tensor")
    rows, cols = info.shape
    if not 0 <= first <= last <= rows:
        raise ValueError(f"{base}: invalid encoded rows [{first},{last})")
    span = (last - first) * (cols // 64 * 36)
    raw = store._payload(info, first * (cols // 64 * 36), first * (cols // 64 * 36) + span)
    codes, scales = _reframe_nvfp4(raw)
    return codes.reshape(last - first, cols // 2), scales.reshape(last - first, cols // 16)


def _sidecar_value(store: GgufSource, sidecar: str, expert: int | None) -> float:
    info = store.describe(sidecar)
    if info.ggml != "F32":
        raise ValueError(f"{sidecar}: expected an F32 scale sidecar")
    flat = _values_flat(store, info, 0, _count(info.shape))
    if expert is None:
        if _count(info.shape) != 1:
            raise ValueError(f"{sidecar}: expected a scalar scale")
        return float(flat[0])
    if len(info.shape) != 1 or info.shape[0] <= expert:
        raise ValueError(f"{sidecar}: expected one scale per expert")
    return float(flat[expert])


def _reciprocal_divisor(store: GgufSource, sidecar: str, expert: int | None) -> bytes:
    from tools.artifact.formats import valid_positive_fp32_word

    value = _sidecar_value(store, sidecar, expert)
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{sidecar}: scale must be finite and positive")
    return struct.pack("<f", 1.0 / value)


def _sidecar_divisor(store: GgufSource, base: str, expert: int | None) -> bytes:
    name = base.removesuffix(".weight") + ".scale"
    if not store.has(name):
        return struct.pack("<f", 1.0)
    return _reciprocal_divisor(store, name, expert)


def _input_divisor(store: GgufSource, base: str, expert: int | None) -> bytes:
    name = base.removesuffix(".weight") + ".input_scale"
    if not store.has(name):
        return struct.pack("<f", 1.0)
    return _reciprocal_divisor(store, name, expert)


def gguf_tensor_source(
    store: GgufSource, hf_name: str, shape: tuple[int, ...]
) -> LogicalSource:
    """Decoded-values source for one GGUF tensor (norms, biases, conv1d)."""
    ref = gguf_translate(hf_name)
    if ref is None or ref.expert is not None:
        raise ValueError(f"{hf_name}: no GGUF tensor mapping")
    info = store.describe(ref.base)
    unit_axis = (
        len(shape) == 3
        and len(info.shape) == 2
        and shape[0] == info.shape[0]
        and shape[1] == 1
        and shape[2] == info.shape[1]
    )
    if tuple(info.shape) != tuple(shape) and not unit_axis:
        raise ValueError(f"{ref.base}: expected source shape {shape}, got {info.shape}")
    role = _role_of(ref.base)

    def read(begin: int, end: int) -> torch.Tensor:
        if role is None:
            flat = _values_flat(store, info, begin, end)
        else:
            full = _ordered_values(store, ref.base, tuple(info.shape))
            flat = full.reshape(shape).reshape(-1)[begin:end]
        return _undo_write_transform(flat, hf_name)

    return LogicalSource(shape, f"{store.path}:{ref.base}", read)


def _count(shape: tuple[int, ...]) -> int:
    count = 1
    for axis in shape:
        count *= axis
    return count


def gguf_matrix_source(
    store: GgufSource, hf_name: str, shape: tuple[int, int], format: str | None = None
) -> LogicalSource:
    """Values or NVFP4-encoded source for one GGUF matrix (lazy NVFP4 resolution)."""
    ref = gguf_translate(hf_name)
    if ref is None or ref.expert is not None:
        raise ValueError(f"{hf_name}: no GGUF matrix mapping")
    info = store.describe(ref.base)
    if tuple(info.shape) != shape:
        raise ValueError(f"{ref.base}: expected matrix shape {shape}, got {info.shape}")
    role = _role_of(ref.base)
    if format is None:
        encoded = info.ggml == "NVFP4"
    elif format == "nvfp4":
        if info.ggml != "NVFP4":
            raise ValueError(f"{ref.base}: cannot serve nvfp4 rows from {info.ggml}")
        encoded = True
    else:
        raise ValueError(
            f"{ref.base}: GGUF serves {format} by decoding values, not encoded rows"
        )

    def read(begin: int, end: int) -> torch.Tensor:
        rows, cols = shape
        if info.ggml != "NVFP4":
            return _matrix_values(store, ref.base, shape, begin, end)
        if role is None:
            first, last = begin // cols, (end + cols - 1) // cols
            codes, scales = _encoded_span(store, ref.base, first, last)
        else:
            codes, scales = _ordered_codes(store, ref.base, shape)
            first, last = begin // cols, (end + cols - 1) // cols
            codes = codes[first:last]
            scales = scales[first:last]
        pairs = torch.stack(
            ((codes & 15).long(), (codes >> 4).long()), dim=-1
        ).reshape(last - first, cols)
        values = _E2M1[pairs] * _ue4m3_to_fp32(scales.repeat_interleave(16, dim=1))
        divisor = struct.unpack("<f", _sidecar_divisor(store, ref.base, None))[0]
        values = values / divisor
        flat = values.reshape(-1)
        return flat[begin - first * cols : end - first * cols]

    def read_encoded(begin: int, end: int) -> EncodedRows:
        if info.ggml != "NVFP4":
            raise ValueError(f"{ref.base}: encoded rows require an NVFP4 tensor")
        rows, cols = shape
        if not 0 <= begin < end <= rows:
            raise ValueError(f"{ref.base}: invalid encoded rows [{begin},{end})")
        if role is None:
            codes, scales = _encoded_span(store, ref.base, begin, end)
        else:
            full_codes, full_scales = _ordered_codes(store, ref.base, shape)
            codes, scales = full_codes[begin:end], full_scales[begin:end]
        return EncodedRows(
            "nvfp4",
            codes,
            scales,
            _sidecar_divisor(store, ref.base, None),
        )

    return LogicalSource(
        shape,
        f"{store.path}:{ref.base}",
        read,
        read_encoded if encoded else None,
        (
            (lambda: _sidecar_divisor(store, ref.base, None))
            if encoded
            else None
        ),
        (
            (lambda: _input_divisor(store, ref.base, None))
            if encoded
            else None
        ),
    )


def _expert_base(store: GgufSource, ref: GgufRef, role: str) -> tuple[str, int | None]:
    """Resolve the stacked expert tensor, falling back to fused gate_up halves."""
    if store.has(ref.base):
        return ref.base, None
    if role in ("gate", "up"):
        fused = ref.base.replace(f"ffn_{role}_exps.weight", "ffn_gate_up_exps.weight")
        if store.has(fused):
            return fused, 0 if role == "gate" else 1
    raise ValueError(f"{store.path}: missing source tensor {ref.base!r}")


def gguf_expert_source(
    store: GgufSource,
    hf_name: str,
    shape: tuple[int, int],
    *,
    expert: int | None = None,
    half: int | None = None,
) -> LogicalSource:
    """One routed expert's matrix from a stacked GGUF expert tensor.

    Per-expert HF names carry the expert index; fused HF names resolve it from
    their offset in ``gguf_param_source`` and pass it explicitly. When the
    separate gate/up banks are absent, gate/up fall back to fused halves.
    """
    ref = gguf_translate(hf_name)
    if ref is None:
        raise ValueError(f"{hf_name}: no GGUF expert mapping")
    if "gate_up_proj" in hf_name:
        role = "gate_up"
    elif ".gate_proj" in hf_name:
        role = "gate"
    elif ".up_proj" in hf_name:
        role = "up"
    else:
        role = "down"
    if expert is None:
        expert = ref.expert
    if expert is None:
        raise ValueError(f"{hf_name}: fused GGUF experts need an expert offset")
    base, fused_half = _expert_base(store, ref, role)
    if half is None:
        half = fused_half
    info = store.describe(base)
    if len(info.shape) != 3:
        raise ValueError(f"{base}: expected a stacked expert tensor")
    experts, rows, cols = info.shape
    if not 0 <= expert < experts:
        raise ValueError(f"{base}: expert {expert} out of range")
    if half is not None:
        if rows % 2:
            raise ValueError(f"{base}: fused experts need even rows")
        if shape != (rows // 2, cols):
            raise ValueError(f"{base}: expected expert shape {shape}")
    elif shape != (rows, cols):
        raise ValueError(f"{base}: expected expert shape {shape}")
    ggml = next(v for v in _GGML_TYPES.values() if v.name == info.ggml)
    stride = rows * cols // ggml.block * ggml.block_bytes
    first_byte, last_byte = expert * stride, (expert + 1) * stride
    encoded = info.ggml == "NVFP4"

    def read(begin: int, end: int) -> torch.Tensor:
        count = shape[0] * shape[1]
        if not 0 <= begin <= end <= count:
            raise ValueError(f"{base}: element range [{begin},{end}) exceeds {shape}")
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        if half is not None:
            full = _expert_rows(info, expert, stride, rows, cols)
            part = full.reshape(rows, cols)[
                half * shape[0] : (half + 1) * shape[0]
            ].reshape(-1)
            return part[begin:end]
        if info.ggml in _PLAIN:
            width = ggml.block_bytes
            raw = store._payload(
                info, first_byte + begin * width, first_byte + end * width
            )
            return _words(raw, _PLAIN[info.ggml])
        first, last = begin // ggml.block, (end + ggml.block - 1) // ggml.block
        raw = store._payload(
            info,
            first_byte + first * ggml.block_bytes,
            first_byte + last * ggml.block_bytes,
        )
        return _DECODERS[info.ggml](raw)[
            begin - first * ggml.block : end - first * ggml.block
        ]

    def _expert_rows(info, expert, stride, rows, cols):
        raw = store._payload(info, expert * stride, (expert + 1) * stride)
        if info.ggml in _PLAIN:
            return _words(raw, _PLAIN[info.ggml])
        return _DECODERS[info.ggml](raw)

    def read_encoded(begin: int, end: int) -> EncodedRows:
        if not encoded:
            raise ValueError(f"{base}: encoded rows require an NVFP4 tensor")
        if not 0 <= begin < end <= shape[0]:
            raise ValueError(f"{base}: invalid encoded rows [{begin},{end})")
        span = (shape[1] // 64) * 36
        if half is not None:
            start = first_byte + half * shape[0] * span
        else:
            start = first_byte
        raw = store._payload(info, start + begin * span, start + end * span)
        codes, scales = _reframe_nvfp4(raw)
        return EncodedRows(
            "nvfp4",
            codes.reshape(end - begin, shape[1] // 2),
            scales.reshape(end - begin, shape[1] // 16),
            _sidecar_divisor(store, base, expert),
        )

    return LogicalSource(
        shape,
        f"{store.path}:{base}[expert {expert}]",
        read,
        read_encoded if encoded else None,
        (lambda: _sidecar_divisor(store, base, expert)) if encoded else None,
        (lambda: _input_divisor(store, base, expert)) if encoded else None,
    )


def gguf_has_experts(store: GgufSource, hf_mlp_prefix: str) -> bool:
    """Whether the GGUF store carries stacked routed experts for one MLP prefix."""
    ref = gguf_translate(hf_mlp_prefix + "experts.0.gate_proj.weight")
    if ref is None:
        return False
    if store.has(ref.base):
        return True
    fused = ref.base.replace("ffn_gate_exps.weight", "ffn_gate_up_exps.weight")
    return store.has(fused)


def gguf_param_source(
    store: GgufSource,
    hf_name: str,
    shape: tuple[int, ...],
    *,
    source_shape: tuple[int, ...],
    offset: int = 0,
    rows: tuple[tuple[int, int], ...] | None = None,
    transpose: tuple[int, ...] | None = None,
    format: str | None = None,
) -> LogicalSource:
    """Build the logical source for one recipe parameter from a GGUF store.

    Mirrors the safetensors branch of ``_Builder.add.factory``: fused HF expert
    tensors select their expert (and gate/up half) by offset, row ranges select
    rows, and transpose wraps a values source.
    """
    ref = gguf_translate(hf_name)
    if ref is None:
        raise ValueError(f"{hf_name}: no GGUF mapping for this HF tensor")
    if transpose is not None:
        if format is not None:
            raise ValueError(f"{hf_name}: transpose source requires value access")
        if rows is not None:
            raise ValueError(f"{hf_name}: rows and transpose do not combine")
        if ref.expert is not None:
            raise ValueError(f"{hf_name}: experts do not transpose")
        return transpose_source(
            gguf_tensor_source(store, hf_name, source_shape), transpose, shape
        )
    if ref.expert is not None:
        if len(shape) != 2:
            raise ValueError(f"{hf_name}: expert mapping requires a matrix")
        base = gguf_expert_source(store, hf_name, shape)
    elif ref.base.endswith("ffn_gate_up_exps.weight") or ref.base.endswith(
        "ffn_down_exps.weight"
    ):
        if len(source_shape) != 3:
            raise ValueError(f"{hf_name}: fused GGUF experts need a 3-D source shape")
        stride = source_shape[1] * source_shape[2]
        expert, remainder = divmod(offset, stride)
        if "gate_up" in ref.base:
            half, inner = divmod(remainder, stride // 2)
            if inner:
                raise ValueError(f"{hf_name}: misaligned fused expert offset")
        else:
            half, inner = None, remainder
            if inner:
                raise ValueError(f"{hf_name}: misaligned fused expert offset")
        expect = (source_shape[1] // (2 if half is not None else 1), source_shape[2])
        if tuple(shape) != expect:
            raise ValueError(f"{hf_name}: fused expert shape disagrees")
        base = gguf_expert_source(store, hf_name, shape, expert=expert, half=half)
    else:
        if offset:
            raise ValueError(f"{hf_name}: GGUF stores no fused HF tensor")
        if rows is not None:
            if len(shape) != 2 or len(source_shape) != 2:
                raise ValueError(f"{hf_name}: row selection requires a matrix")
            return select_rows(
                gguf_matrix_source(store, hf_name, source_shape, format), rows
            )
        if len(shape) == 2:
            base = gguf_matrix_source(store, hf_name, shape, format)
        else:
            if format is not None:
                raise ValueError(f"{hf_name}: transpose source requires value access")
            base = gguf_tensor_source(store, hf_name, shape)
    if rows is not None:
        if len(shape) != 2:
            raise ValueError(f"{hf_name}: row selection requires a matrix")
        return select_rows(base, rows)
    return base
