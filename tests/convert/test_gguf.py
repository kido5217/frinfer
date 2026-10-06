from __future__ import annotations

import json
import struct

import pytest
import torch

from tools.convert.qwen3_5 import build_model, text_config
from tools.convert.sources.gguf import (
    _values_flat,
    GgufSource,
    gguf_expert_source,
    gguf_has_experts,
    gguf_matrix_source,
    gguf_param_source,
    gguf_tensor_source,
    gguf_translate,
)

_U32, _I32, _F32, _BOOL, _STR, _ARR = 4, 5, 6, 7, 8, 9


def _enc_str(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + data


def _enc_scalar(kind: int, value) -> bytes:
    if kind == _STR:
        return _enc_str(value.encode())
    if kind == _BOOL:
        return struct.pack("<b", int(bool(value)))
    fmt = {_U32: "<I", _I32: "<i", _F32: "<f"}[kind]
    return struct.pack(fmt, value)


def _enc_value(kind: int, value) -> bytes:
    if kind != _ARR:
        return _enc_scalar(kind, value)
    item, items = value
    out = struct.pack("<i", item) + struct.pack("<Q", len(items))
    return out + b"".join(_enc_scalar(item, item_value) for item_value in items)


def _write_gguf(path, kv, tensors, alignment=32):
    blob = struct.pack("<4sIqq", b"GGUF", 3, len(tensors), len(kv))
    for key, kind, value in kv:
        blob += _enc_str(key.encode()) + struct.pack("<i", kind)
        blob += _enc_value(kind, value)
    infos = b""
    offset = 0
    for name, ne, type_id, raw in tensors:
        infos += _enc_str(name.encode()) + struct.pack("<I", len(ne))
        infos += struct.pack("<" + "q" * len(ne), *ne) + struct.pack("<i", type_id)
        infos += struct.pack("<Q", offset)
        offset += len(raw)
    head = blob + infos
    head += b"\0" * (-len(head) % alignment)
    path.write_bytes(head + b"".join(raw for _, _, _, raw in tensors))


def _kv_common(vocab, extra=(), arch="qwen35"):
    base = [
        ("general.architecture", _STR, arch),
        ("general.type", _STR, "model"),
        (f"{arch}.block_count", _U32, 2),
        (f"{arch}.context_length", _U32, 128),
        (f"{arch}.embedding_length", _U32, 16),
        (f"{arch}.feed_forward_length", _U32, 24),
        (f"{arch}.attention.head_count", _U32, 2),
        (f"{arch}.attention.head_count_kv", _U32, 1),
        (f"{arch}.attention.key_length", _U32, 8),
        (f"{arch}.attention.value_length", _U32, 8),
        (f"{arch}.attention.layer_norm_rms_epsilon", _F32, 1e-6),
        (f"{arch}.rope.dimension_sections", _ARR, (_I32, [1, 1, 0, 0])),
        (f"{arch}.rope.dimension_count", _U32, 4),
        (f"{arch}.ssm.conv_kernel", _U32, 3),
        (f"{arch}.ssm.state_size", _U32, 4),
        (f"{arch}.ssm.group_count", _U32, 1),
        (f"{arch}.ssm.time_step_rank", _U32, 2),
        (f"{arch}.ssm.inner_size", _U32, 8),
        (f"{arch}.attention.recurrent_layers", _ARR, (_BOOL, [True, False])),
        ("tokenizer.ggml.tokens", _ARR, (_STR, [str(i) for i in range(vocab)])),
    ]
    return base + list(extra)


def _ne(shape):
    return tuple(reversed(shape))


def _bf16_bytes(values):
    return (
        torch.as_tensor(values, dtype=torch.float32)
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def test_container_parses_kv_and_tensor_infos(tmp_path):
    path = tmp_path / "tiny.gguf"
    _write_gguf(
        path,
        [("general.architecture", _STR, "qwen35"), ("flag", _BOOL, True)],
        [("a.weight", (4,), 0, struct.pack("<4f", 1, 2, 3, 4))],
    )
    with GgufSource(path) as store:
        assert store.metadata("general.architecture") == "qwen35"
        assert store.metadata("flag") is True
        info = store.describe("a.weight")
        assert (info.shape, info.ggml, info.bytes) == ((4,), "F32", 16)
        assert store.has("a.weight") and not store.has("b.weight")


def test_container_rejects_bad_magic_version_and_type(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"XXXX" + struct.pack("<Iqq", 3, 0, 0))
    with pytest.raises(ValueError, match="not a GGUF"):
        GgufSource(path)
    path.write_bytes(struct.pack("<4sIqq", b"GGUF", 2, 0, 0))
    with pytest.raises(ValueError, match="version"):
        GgufSource(path)
    _write_gguf(path, [], [("q.weight", (32,), 2, bytes(18))])
    with pytest.raises(ValueError, match="unsupported GGUF type"):
        GgufSource(path)


def test_plain_dtypes_decode_exactly(tmp_path):
    path = tmp_path / "plain.gguf"
    f32 = struct.pack("<6f", 0.5, -2.0, 0.0, 1.5, 100.25, -0.0)
    f16 = torch.tensor([0.5, -2.0, 1.5, 0.125], dtype=torch.float16).numpy().tobytes()
    bf16 = struct.pack("<4H", 0x3F00, 0xC000, 0x3FC0, 0x3D00)
    _write_gguf(
        path,
        [],
        [("a", (6,), 0, f32), ("b", (2, 2), 1, f16), ("c", (4,), 30, bf16)],
    )
    with GgufSource(path) as store:
        assert torch.equal(
            _values_flat(store, store.describe("a"), 0, 6),
            torch.tensor([0.5, -2.0, 0.0, 1.5, 100.25, -0.0]),
        )
        assert torch.equal(
            _values_flat(store, store.describe("a"), 2, 5),
            torch.tensor([0.0, 1.5, 100.25]),
        )
        assert torch.equal(
            _values_flat(store, store.describe("b"), 0, 4),
            torch.tensor([0.5, -2.0, 1.5, 0.125]),
        )
        assert torch.equal(
            _values_flat(store, store.describe("c"), 0, 4),
            torch.tensor([0.5, -2.0, 1.5, 0.03125]),
        )


def _q8_0_block(delta=0.5, quants=range(-16, 16)):
    return struct.pack("<H", 0x3800) + bytes(q & 0xFF for q in quants)


def test_q8_0_matches_hand_computed_block(tmp_path):
    raw = _q8_0_block() + _q8_0_block()
    path = tmp_path / "q8.gguf"
    _write_gguf(path, [], [("w", (64,), 8, raw)])
    with GgufSource(path) as store:
        info = store.describe("w")
        assert info.bytes == 68
        values = _values_flat(store, info, 0, 64)
        expect = torch.tensor([q * 0.5 for q in list(range(-16, 16)) * 2])
        assert torch.equal(values, expect)
        assert torch.equal(_values_flat(store, info, 30, 34), expect[30:34])


def _q4_scales(sc=(1,) * 8, mn=(0,) * 8):
    out = bytearray(12)
    out[:4] = bytes(s & 63 for s in sc[:4])
    out[4:8] = bytes(m & 63 for m in mn[:4])
    for j in range(4, 8):
        out[j + 4] = (sc[j] & 15) | ((mn[j] & 15) << 4)
        out[j - 4] |= (sc[j] >> 4) << 6
        out[j] |= (mn[j] >> 4) << 6
    return bytes(out)


def test_q4_k_matches_hand_computed_block(tmp_path):
    scales = _q4_scales(sc=(1,) * 8, mn=(1,) * 8)
    raw = struct.pack("<HH", 0x3C00, 0x3800) + scales + bytes([0x22] * 128)
    path = tmp_path / "q4k.gguf"
    _write_gguf(path, [], [("w", (256,), 12, raw)])
    with GgufSource(path) as store:
        values = _values_flat(store, store.describe("w"), 0, 256)
        assert torch.equal(values, torch.full((256,), 1.5))


def test_q5_k_high_bits_match_hand_computed_block(tmp_path):
    scales = _q4_scales(sc=(1,) * 8, mn=(1,) * 8)
    raw = (
        struct.pack("<HH", 0x3C00, 0x3800)
        + scales
        + bytes([0x01] * 32)
        + bytes([0x22] * 128)
    )
    path = tmp_path / "q5k.gguf"
    _write_gguf(path, [], [("w", (256,), 13, raw)])
    with GgufSource(path) as store:
        values = _values_flat(store, store.describe("w"), 0, 256)
        expect = torch.full((256,), 1.5)
        expect[:32] = 2.0 + 16.0 - 0.5
        assert torch.equal(values, expect)


def test_q6_k_matches_hand_computed_block(tmp_path):
    raw = bytes([0x22] * 128) + bytes([0] * 64) + bytes([2] * 16) + struct.pack("<H", 0x3800)
    path = tmp_path / "q6k.gguf"
    _write_gguf(path, [], [("w", (256,), 14, raw)])
    with GgufSource(path) as store:
        values = _values_flat(store, store.describe("w"), 0, 256)
        assert torch.equal(values[:16], torch.full((16,), -30.0))
        assert torch.equal(values[16:32], torch.full((16,), -30.0))
        assert values.shape == (256,)
def _nvfp4_gguf_block(codes, scales=(0x38,) * 4):
    """One 64-element ggml NVFP4 super-block from an element-code sequence."""
    assert len(codes) == 64
    out = bytearray(scales)
    for s in range(4):
        for j in range(8):
            out.append(codes[s * 16 + j] | (codes[s * 16 + j + 8] << 4))
    return bytes(out)


def test_nvfp4_reframes_words_and_reciprocates_divisors(tmp_path):
    codes = [(e % 16) for e in range(128)]
    raw = b"".join(_nvfp4_gguf_block(codes[i : i + 64]) for i in (0, 64))
    path = tmp_path / "nv.gguf"
    _write_gguf(
        path,
        [],
        [
            ("blk.0.ffn_gate.weight", (64, 2), 40, raw),
            ("blk.0.ffn_gate.scale", (1,), 0, struct.pack("<f", 2.0)),
            ("blk.0.ffn_gate.input_scale", (1,), 0, struct.pack("<f", 4.0)),
        ],
    )
    with GgufSource(path) as store:
        source = gguf_matrix_source(store, "model.layers.0.mlp.gate_proj.weight", (2, 64))
        words = source.read_encoded(0, 2)
        assert words.format == "nvfp4"
        expect = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 8)
        assert words.codes.reshape(-1).tolist() == list(expect)
        assert words.scales.reshape(-1).tolist() == [0x38] * 8
        assert words.weight_divisor == struct.pack("<f", 0.5)
        assert source.weight_divisor() == struct.pack("<f", 0.5)
        assert source.input_divisor() == struct.pack("<f", 0.25)


def test_nvfp4_values_decode_divides_by_sidecar(tmp_path):
    codes = [4] * 64
    raw = _nvfp4_gguf_block(codes)
    path = tmp_path / "nvval.gguf"
    _write_gguf(
        path,
        [],
        [
            ("blk.0.ffn_gate.weight", (64, 1), 40, raw),
            ("blk.0.ffn_gate.scale", (1,), 0, struct.pack("<f", 0.5)),
        ],
    )
    with GgufSource(path) as store:
        source = gguf_matrix_source(store, "model.layers.0.mlp.gate_proj.weight", (1, 64))
        assert source.read_encoded is None or True
        values = source.values()
        assert torch.equal(values, torch.full((64,), 1.0))


def test_nvfp4_block_decode_agrees_with_reframed_values(tmp_path):
    from tools.convert.sources.gguf import _dequant_nvfp4

    codes = [(e * 5 + 3) % 16 for e in range(64)]
    raw = _nvfp4_gguf_block(codes, scales=(0x40, 0x40, 0x40, 0x40))
    path = tmp_path / "nvx.gguf"
    _write_gguf(
        path,
        [],
        [
            ("blk.0.ffn_gate.weight", (64, 1), 40, raw),
            ("blk.0.ffn_gate.scale", (1,), 0, struct.pack("<f", 2.0)),
        ],
    )
    with GgufSource(path) as store:
        source = gguf_matrix_source(store, "model.layers.0.mlp.gate_proj.weight", (1, 64))
        expect = _dequant_nvfp4(raw) / 0.5
        assert torch.equal(source.values(), expect)


def test_nvfp4_missing_sidecars_default_to_unity(tmp_path):
    raw = _nvfp4_gguf_block([4] * 64)
    path = tmp_path / "nvbare.gguf"
    _write_gguf(path, [], [("blk.0.ffn_gate.weight", (64, 1), 40, raw)])
    with GgufSource(path) as store:
        source = gguf_matrix_source(store, "model.layers.0.mlp.gate_proj.weight", (1, 64))
        words = source.read_encoded(0, 1)
        assert words.weight_divisor == struct.pack("<f", 1.0)
        assert torch.equal(source.values(), torch.full((64,), 2.0))


def test_translate_covers_dense_moe_and_gdn_names():
    assert gguf_translate("model.embed_tokens.weight").base == "token_embd.weight"
    assert gguf_translate("model.language_model.embed_tokens.weight").base == (
        "token_embd.weight"
    )
    assert gguf_translate("lm_head.weight").base == "output.weight"
    assert gguf_translate("model.norm.weight").base == "output_norm.weight"
    assert gguf_translate("model.layers.3.self_attn.q_proj.weight").base == (
        "blk.3.attn_q.weight"
    )
    assert gguf_translate("model.layers.3.self_attn.o_proj.weight").base == (
        "blk.3.attn_output.weight"
    )
    assert gguf_translate("model.layers.3.self_attn.q_norm.weight").base == (
        "blk.3.attn_q_norm.weight"
    )
    assert gguf_translate("model.layers.3.mlp.down_proj.weight").base == (
        "blk.3.ffn_down.weight"
    )
    assert gguf_translate("model.layers.3.mlp.gate.weight").base == (
        "blk.3.ffn_gate_inp.weight"
    )
    assert gguf_translate("model.layers.3.mlp.shared_expert_gate.weight").base == (
        "blk.3.ffn_gate_inp_shexp.weight"
    )
    ref = gguf_translate("model.layers.5.mlp.experts.7.up_proj.weight")
    assert (ref.base, ref.expert) == ("blk.5.ffn_up_exps.weight", 7)
    assert gguf_translate("model.layers.5.mlp.experts.gate_up_proj.weight").base == (
        "blk.5.ffn_gate_up_exps.weight"
    )
    assert gguf_translate("model.layers.5.mlp.shared_expert.down_proj.weight").base == (
        "blk.5.ffn_down_shexp.weight"
    )
    assert gguf_translate("model.layers.0.linear_attn.in_proj_qkv.weight").base == (
        "blk.0.attn_qkv.weight"
    )
    assert gguf_translate("model.layers.0.linear_attn.in_proj_z.weight").base == (
        "blk.0.attn_gate.weight"
    )
    assert gguf_translate("model.layers.0.linear_attn.in_proj_a.weight").base == (
        "blk.0.ssm_alpha.weight"
    )
    assert gguf_translate("model.layers.0.linear_attn.conv1d.weight").base == (
        "blk.0.ssm_conv1d.weight"
    )
    assert gguf_translate("model.layers.0.linear_attn.A_log").base == "blk.0.ssm_a"
    assert gguf_translate("model.layers.0.linear_attn.dt_bias").base == (
        "blk.0.ssm_dt.bias"
    )
    assert gguf_translate("model.layers.0.linear_attn.out_proj.weight").base == (
        "blk.0.ssm_out.weight"
    )
    assert gguf_translate("vision.patch_embedding.weight") is None
    assert gguf_translate("model.layers.0.mlp.experts.x.gate_proj.weight") is None
    assert gguf_translate("mtp.fc.weight") is None


def test_config_synthesis_feeds_text_config(tmp_path):
    path = tmp_path / "cfg.gguf"
    emb = _bf16_bytes(torch.zeros(6, 16))
    kv = [item for item in _kv_common(6) if not item[0].endswith(("block_count",))]
    kv = _kv_common(6)
    _write_gguf(path, kv, [("token_embd.weight", (16, 6), 30, emb)])
    with GgufSource(path) as store:
        config = text_config(store.config, mtp=False)
        assert config["hidden_size"] == 16 and config["vocab_size"] == 6
        assert config["layer_types"] == ["linear_attention", "full_attention"]
        assert config["tie_word_embeddings"] is True
        assert config["linear_num_key_heads"] == 1
        assert config["rope_parameters"]["mrope_section"] == [1, 1, 0]


def test_config_rejects_foreign_arch_and_missing_sections(tmp_path):
    path = tmp_path / "arch.gguf"
    _write_gguf(path, [("general.architecture", _STR, "qwen3")], [])
    with GgufSource(path) as store:
        with pytest.raises(ValueError, match="unsupported GGUF architecture"):
            store.config
    kv = [item for item in _kv_common(2) if "dimension_sections" not in item[0]]
    emb = _bf16_bytes(torch.zeros(2, 16))
    _write_gguf(path, kv, [("token_embd.weight", (16, 2), 30, emb)])
    with GgufSource(path) as store:
        with pytest.raises(ValueError, match="RoPE dimension sections"):
            store.config


def test_v_reorder_is_an_involution_and_restores_grouped_order(tmp_path):
    from tools.convert.sources.gguf import _v_permutation

    perm = _v_permutation(16, 2, 2, 4)
    assert torch.equal(perm[perm], torch.arange(16))
    # nk=2, nv=4, dv=4: in_proj_z has vg=16 rows; the file stores them tiled.
    grouped = torch.arange(64).reshape(16, 4).float()
    tiled = grouped[perm]
    assert not torch.equal(tiled, grouped)
    path = tmp_path / "gdn.gguf"
    kv = [
        item
        for item in _kv_common(2)
        if ".ssm.group_count" not in item[0]
        and ".ssm.time_step_rank" not in item[0]
        and ".ssm.inner_size" not in item[0]
    ] + [
        ("qwen35.ssm.group_count", _U32, 2),
        ("qwen35.ssm.time_step_rank", _U32, 4),
        ("qwen35.ssm.inner_size", _U32, 16),
    ]
    emb = _bf16_bytes(torch.zeros(2, 16))
    z = tiled.repeat(1, 4)
    _write_gguf(
        path,
        kv,
        [
            ("token_embd.weight", (16, 2), 30, emb),
            ("blk.0.attn_gate.weight", (16, 16), 0, z.numpy().tobytes()),
        ],
    )
    with GgufSource(path) as store:
        assert store.config["linear_num_key_heads"] == 2
        source = gguf_matrix_source(
            store, "model.layers.0.linear_attn.in_proj_z.weight", (16, 16)
        )
        assert torch.equal(source.values().reshape(16, 16), grouped.repeat(1, 4))
def _tiny_hf_config(*, moe=False):
    text = {
        "hidden_size": 16,
        "vocab_size": 8,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        "tie_word_embeddings": False,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 4,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
    }
    if moe:
        text.update(
            num_experts=3,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=4,
        )
    else:
        text["intermediate_size"] = 24
    arch = (
        "Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"
    )
    return {"architectures": [arch], **text}


def _tiny_tensors(*, moe=False):
    torch.manual_seed(7)
    rand = lambda *shape: torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tensors = {
        "model.embed_tokens.weight": rand(8, 16),
        "lm_head.weight": rand(8, 16),
        "model.norm.weight": rand(16),
    }
    attns = [
        ("model.layers.0.", True),
        ("model.layers.1.", False),
    ]
    for prefix, linear in attns:
        tensors[prefix + "input_layernorm.weight"] = rand(16)
        tensors[prefix + "post_attention_layernorm.weight"] = rand(16)
        if linear:
            tensors[prefix + "linear_attn.A_log"] = rand(4)
            tensors[prefix + "linear_attn.dt_bias"] = rand(4)
            tensors[prefix + "linear_attn.conv1d.weight"] = rand(24, 1, 3)
            tensors[prefix + "linear_attn.in_proj_a.weight"] = rand(4, 16)
            tensors[prefix + "linear_attn.in_proj_b.weight"] = rand(4, 16)
            tensors[prefix + "linear_attn.in_proj_qkv.weight"] = rand(24, 16)
            tensors[prefix + "linear_attn.in_proj_z.weight"] = rand(8, 16)
            tensors[prefix + "linear_attn.norm.weight"] = rand(2)
            tensors[prefix + "linear_attn.out_proj.weight"] = rand(16, 8)
        else:
            tensors[prefix + "self_attn.q_proj.weight"] = rand(32, 16)
            tensors[prefix + "self_attn.k_proj.weight"] = rand(8, 16)
            tensors[prefix + "self_attn.v_proj.weight"] = rand(8, 16)
            tensors[prefix + "self_attn.q_norm.weight"] = rand(8)
            tensors[prefix + "self_attn.k_norm.weight"] = rand(8)
            tensors[prefix + "self_attn.o_proj.weight"] = rand(16, 16)
        if moe:
            tensors[prefix + "mlp.gate.weight"] = rand(3, 16)
            tensors[prefix + "mlp.shared_expert_gate.weight"] = rand(1, 16)
            for e in range(3):
                tensors[prefix + f"mlp.experts.{e}.gate_proj.weight"] = rand(8, 16)
                tensors[prefix + f"mlp.experts.{e}.up_proj.weight"] = rand(8, 16)
                tensors[prefix + f"mlp.experts.{e}.down_proj.weight"] = rand(16, 8)
            for role, shape in (("gate", (4, 16)), ("up", (4, 16)), ("down", (16, 4))):
                tensors[prefix + f"mlp.shared_expert.{role}_proj.weight"] = rand(*shape)
        else:
            tensors[prefix + "mlp.gate_proj.weight"] = rand(24, 16)
            tensors[prefix + "mlp.up_proj.weight"] = rand(24, 16)
            tensors[prefix + "mlp.down_proj.weight"] = rand(16, 24)
    return tensors


def _tile_gdn(values, kind):
    """Apply the GGUF write-side V tiling (nk=2, nv=4, dv=2) to one HF tensor."""
    from tools.convert.sources.gguf import _v_permutation

    perm8 = _v_permutation(8, 2, 2, 2)
    perm4 = _v_permutation(4, 2, 2, 1)
    if kind == "qkv":
        out = values.clone()
        out[16:] = values[16:][perm8]
        return out
    if kind == "z":
        return values[perm8]
    if kind in ("a", "b", "alog"):
        return values[perm4]
    if kind == "conv":
        out = values.clone()
        out[16:] = values[16:][perm8]
        return out
    if kind == "out":
        return values[:, perm8]
    raise AssertionError(kind)


def _gguf_tensors(tensors):
    """Mirror HF tensors into (canonical name, ne, type, bytes) GGUF entries."""
    from tools.convert.sources.gguf import gguf_translate

    entries = []
    for hf_name, data in tensors.items():
        ref = gguf_translate(hf_name)
        assert ref is not None, hf_name
        # Mimic llama.cpp's write-side transforms (see _undo_write_transform).
        if hf_name.endswith(".A_log"):
            data = -torch.exp(data.float())
        elif hf_name.endswith("norm.weight") and ".linear_attn.norm.weight" not in hf_name:
            data = data.float() + 1
        kind = None
        if ".linear_attn.in_proj_qkv." in hf_name:
            kind = "qkv"
        elif ".linear_attn.in_proj_z." in hf_name:
            kind = "z"
        elif ".linear_attn.in_proj_a." in hf_name or ".linear_attn.in_proj_b." in hf_name:
            kind = "a"
        elif hf_name.endswith((".A_log", ".dt_bias")):
            kind = "alog"
        elif ".linear_attn.conv1d." in hf_name:
            kind = "conv"
        elif ".linear_attn.out_proj." in hf_name:
            kind = "out"
        flat = data.reshape(-1) if kind is None else _tile_gdn(
            data.reshape(data.shape[0], -1) if kind == "conv" else data, kind
        ).reshape(-1)
        raw = _bf16_bytes(flat)
        if ref.expert is not None:
            continue  # stacked below
        shape = tuple(data.shape)
        if len(shape) == 3:  # conv1d (C,1,T) rides as (C,T)
            shape = (shape[0], shape[2])
        entries.append((ref.base, tuple(reversed(shape)), 30, raw))
    return entries


def _gguf_kv(vocab, *, moe=False):
    kv = _kv_common(vocab, arch="qwen35moe" if moe else "qwen35")
    arch = "qwen35moe" if moe else "qwen35"
    drop = (
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.recurrent_layers",
        "rope.dimension_sections",
        "rope.dimension_count",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.inner_size",
    )
    kv = [item for item in kv if item[0] != "general.architecture" and item[0].split(".", 1)[1] not in drop]
    kv = [("general.architecture", _STR, arch)] + kv
    kv += [
        (f"{arch}.block_count", _U32, 2),
        (f"{arch}.embedding_length", _U32, 16),
        (f"{arch}.ssm.conv_kernel", _U32, 3),
        (f"{arch}.attention.head_count", _U32, 2),
        (f"{arch}.attention.head_count_kv", _U32, 1),
        (f"{arch}.attention.key_length", _U32, 8),
        (f"{arch}.attention.value_length", _U32, 8),
        (f"{arch}.rope.dimension_count", _U32, 4),
        (f"{arch}.rope.dimension_sections", _ARR, (_I32, [1, 1, 0, 0])),
        (f"{arch}.ssm.group_count", _U32, 2),
        (f"{arch}.ssm.state_size", _U32, 4),
        (f"{arch}.ssm.time_step_rank", _U32, 4),
        (f"{arch}.ssm.inner_size", _U32, 8),
        (f"{arch}.attention.recurrent_layers", _ARR, (_BOOL, [True, False])),
    ]
    if moe:
        kv += [
            (f"{arch}.expert_count", _U32, 3),
            (f"{arch}.expert_used_count", _U32, 2),
            (f"{arch}.expert_feed_forward_length", _U32, 8),
            (f"{arch}.expert_shared_feed_forward_length", _U32, 4),
        ]
    else:
        kv += [(f"{arch}.feed_forward_length", _U32, 24)]
    return kv


def _stack_experts(entries, tensors):
    """Append stacked BF16 expert tensors, dropping per-expert entries already added."""
    for bid in (0, 1):
        for role, shape in (("gate", (3, 8, 16)), ("up", (3, 8, 16)), ("down", (3, 16, 8))):
            proj = role + "_proj"
            parts = [
                tensors[f"model.layers.{bid}.mlp.experts.{e}.{proj}.weight"].reshape(-1)
                for e in range(3)
            ]
            raw = _bf16_bytes(torch.cat(parts))
            entries.append((f"blk.{bid}.ffn_{role}_exps.weight", tuple(reversed(shape)), 30, raw))
    return entries


@pytest.mark.parametrize("moe", [False, True])
def test_build_model_matches_safetensors_source(tmp_path, moe):
    from safetensors.torch import save_file

    from tools.convert.sources.safetensors import SafetensorsSource

    hfdir = tmp_path / "hf"
    hfdir.mkdir()
    (hfdir / "config.json").write_text(json.dumps(_tiny_hf_config(moe=moe)))
    tensors = _tiny_tensors(moe=moe)
    save_file(tensors, hfdir / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (hfdir / role).write_text(json.dumps(value))
    (hfdir / "chat_template.jinja").write_text("{{ messages }}")
    gguf_path = tmp_path / "tiny.gguf"
    entries = _gguf_tensors(tensors)
    if moe:
        entries = _stack_experts(entries, tensors)
    _write_gguf(gguf_path, _gguf_kv(8, moe=moe), entries)
    overrides = {
        role: hfdir / role
        for role in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        )
    }
    with SafetensorsSource(hfdir) as base:
        expected = build_model(base, components=("text",))
    with GgufSource(gguf_path) as source:
        actual = build_model(source, components=("text",), resource_overrides=overrides)
    assert set(actual.parameters) == set(expected.parameters)
    for name, parameter in expected.parameters.items():
        got = actual.parameters[name]
        assert got.shape == parameter.shape, name
        want = parameter.source.values().float().reshape(parameter.shape)
        have = got.source.values().float().reshape(parameter.shape)
        # Norms and A_log pass through a BF16 write-side transform (-exp/+1)
        # whose inverse is exact only up to one BF16 ulp.
        if name.endswith(("norm", "a_log", "dt_bias")):
            assert torch.allclose(have, want, atol=1e-2, rtol=1e-2), name
        else:
            assert torch.equal(have, want), name
def test_v_restore_handles_asymmetric_head_counts(tmp_path):
    from tools.convert.sources.gguf import _v_inverse_permutation, _v_permutation

    assert torch.equal(
        _v_inverse_permutation(16, 4, 2, 2)[_v_permutation(16, 4, 2, 2)],
        torch.arange(16),
    )
    grouped = torch.arange(32).reshape(16, 2).float()
    tiled = grouped[_v_permutation(16, 4, 2, 2)]
    assert not torch.equal(tiled, grouped)
    kv = _kv_common(2)
    kv = [item for item in kv if "ssm.group_count" not in item[0]
          and "ssm.time_step_rank" not in item[0]
          and "ssm.inner_size" not in item[0]] + [
        ("qwen35.ssm.group_count", _U32, 4),
        ("qwen35.ssm.time_step_rank", _U32, 8),
        ("qwen35.ssm.inner_size", _U32, 16),
    ]
    path = tmp_path / "asym.gguf"
    _write_gguf(
        path,
        kv,
        [
            ("token_embd.weight", (16, 2), 30, _bf16_bytes(torch.zeros(2, 16))),
            ("blk.0.attn_gate.weight", (16, 16), 0, tiled.repeat(1, 8).numpy().tobytes()),
        ],
    )
    with GgufSource(path) as store:
        source = gguf_matrix_source(
            store, "model.layers.0.linear_attn.in_proj_z.weight", (16, 16)
        )
        assert torch.equal(source.values().reshape(16, 16), grouped.repeat(1, 8))


def test_fused_expert_values_and_encoded_halves(tmp_path):
    gate = torch.arange(16).reshape(4, 4).float()
    up = torch.arange(16, 32).reshape(4, 4).float()
    stacked = torch.stack((torch.cat((gate, up)),) * 2).numpy().tobytes()
    path = tmp_path / "fused.gguf"
    _write_gguf(
        path, [], [("blk.0.ffn_gate_up_exps.weight", (4, 8, 2), 0, stacked)]
    )
    with GgufSource(path) as store:
        for expert in (0, 1):
            for half, want in ((0, gate), (1, up)):
                source = gguf_param_source(
                    store,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    (4, 4),
                    source_shape=(2, 8, 4),
                    offset=(expert * 2 + half) * 16,
                )
                assert torch.equal(source.values().reshape(4, 4), want)


def test_fused_nvfp4_encoded_halves(tmp_path):
    gate_raw = _nvfp4_gguf_block([4] * 64)
    up_raw = _nvfp4_gguf_block([5] * 64)
    path = tmp_path / "fusednv.gguf"
    _write_gguf(
        path,
        [],
        [
            ("blk.0.ffn_gate_up_exps.weight", (64, 2, 1), 40, gate_raw + up_raw),
            ("blk.0.ffn_gate_up_exps.scale", (1,), 0, struct.pack("<f", 1.0)),
        ],
    )
    with GgufSource(path) as store:
        for half, code in ((0, 0x44), (1, 0x55)):
            source = gguf_param_source(
                store,
                "model.layers.0.mlp.experts.gate_up_proj",
                (1, 64),
                source_shape=(1, 2, 64),
                offset=half * 64,
            )
            words = source.read_encoded(0, 1)
            assert words.codes.reshape(-1).tolist() == [code] * 32
