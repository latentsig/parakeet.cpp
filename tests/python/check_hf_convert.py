#!/usr/bin/env python3
"""Model-independent checks for scripts/convert_hf_parakeet_to_gguf.py."""
import importlib.util
import pathlib
import sys

try:
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "hfconv",
        pathlib.Path(__file__).resolve().parents[2] / "scripts" / "convert_hf_parakeet_to_gguf.py")
    hfconv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hfconv)
except SystemExit:
    sys.exit(77)  # gguf missing
except ImportError:
    sys.exit(77)


def pack(codes):
    """Inverse of unpack_ternary: codes [out, in] in {0,1,2} -> uint8 [out, ceil(in/5)]."""
    out, n = codes.shape
    nb = -(-n // 5)
    pad = np.zeros((out, nb * 5), dtype=np.int64)
    pad[:, :n] = codes
    return (pad.reshape(out, nb, 5) * (3 ** np.arange(5))).sum(-1).astype(np.uint8)


def test_unpack_matches_loop(n_in, group=128, out=7, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(out, n_in))
    ng = -(-n_in // group)
    scales = rng.random((out, ng)).astype(np.float16)
    got = hfconv.unpack_ternary(pack(codes), scales, n_in, group)
    want = np.zeros((out, n_in), dtype=np.float32)
    for r in range(out):
        for c in range(n_in):
            want[r, c] = np.float32(scales[r, c // group]) * (codes[r, c] - 1)
    assert got.shape == (out, n_in) and got.dtype == np.float32
    assert np.array_equal(got, want), f"unpack mismatch for in={n_in}"


def test_unpack_rejects_bad_byte():
    q = np.array([[243]], dtype=np.uint8)  # 3**5, not a valid 5-trit byte
    try:
        hfconv.unpack_ternary(q, np.ones((1, 1), dtype=np.float16), 5, 128)
    except ValueError:
        return
    raise AssertionError("byte 243 must be rejected")


def test_packed_byte_validity():
    """The converter and the C++ loader accept exactly the bytes 0..242."""
    hfconv.check_packed_bytes(np.array([[0, 121, 242]], dtype=np.uint8))
    for bad in (243, 250, 255):
        q = np.array([[0, 0, 7], [1, bad, 2]], dtype=np.uint8)
        for fn in (hfconv.check_packed_bytes,
                   lambda x: hfconv.unpack_ternary(x, np.ones((2, 1), dtype=np.float16), 15, 128)):
            try:
                fn(q)
            except ValueError as e:
                assert str(bad) in str(e), e
                continue
            raise AssertionError(f"byte {bad} must be rejected")


RENAMES = {
    "encoder.subsampling.layers.3.weight": "encoder.pre_encode.conv.3.weight",
    "encoder.subsampling.linear.bias": "encoder.pre_encode.out.bias",
    "encoder.layers.5.conv.norm.running_var": "encoder.layers.5.conv.batch_norm.running_var",
    "encoder.layers.5.self_attn.q_proj.weight": "encoder.layers.5.self_attn.linear_q.weight",
    "encoder.layers.5.self_attn.k_proj.weight": "encoder.layers.5.self_attn.linear_k.weight",
    "encoder.layers.5.self_attn.v_proj.weight": "encoder.layers.5.self_attn.linear_v.weight",
    "encoder.layers.5.self_attn.o_proj.weight": "encoder.layers.5.self_attn.linear_out.weight",
    "encoder.layers.5.self_attn.relative_k_proj.weight": "encoder.layers.5.self_attn.linear_pos.weight",
    "encoder.layers.5.self_attn.bias_u": "encoder.layers.5.self_attn.pos_bias_u",
    "encoder.layers.23.self_attn.bias_v": "encoder.layers.23.self_attn.pos_bias_v",
    "decoder.embedding.weight": "decoder.prediction.embed.weight",
    "decoder.lstm.weight_ih_l1": "decoder.prediction.dec_rnn.lstm.weight_ih_l1",
    "encoder_projector.weight": "joint.enc.weight",
    "decoder.decoder_projector.bias": "joint.pred.bias",
    "joint.head.weight": "joint.joint_net.2.weight",
    "encoder.layers.0.feed_forward1.linear1.weight": "encoder.layers.0.feed_forward1.linear1.weight",
    "encoder.layers.0.norm_out.bias": "encoder.layers.0.norm_out.bias",
    "encoder.layers.0.conv.pointwise_conv1.weight": "encoder.layers.0.conv.pointwise_conv1.weight",
}


def test_renames():
    for hf, nemo in RENAMES.items():
        assert hfconv.hf_to_nemo(hf) == nemo, (hf, hfconv.hf_to_nemo(hf), nemo)
    try:
        hfconv.hf_to_nemo("something.unknown")
    except KeyError:
        return
    raise AssertionError("unknown tensor must raise KeyError")


def test_keep_matches_dequant():
    """Needs a real Redux download; skipped unless PK_HF_REDUX_DIR and PK_HF_TEMPLATE are set."""
    import os
    import subprocess
    import tempfile
    d, t = os.environ.get("PK_HF_REDUX_DIR"), os.environ.get("PK_HF_TEMPLATE")
    if not d or not t:
        print("check_hf_convert: keep-vs-dequant skipped (set PK_HF_REDUX_DIR, PK_HF_TEMPLATE)")
        return
    import gguf
    script = str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "convert_hf_parakeet_to_gguf.py")
    with tempfile.TemporaryDirectory() as tmp:
        keep, deq = tmp + "/keep.gguf", tmp + "/deq.gguf"
        common = [sys.executable, script, "--hf", d, "--template", t, "--dtype", "f32"]
        subprocess.run(common + ["--output", keep, "--ternary", "keep"], check=True)
        subprocess.run(common + ["--output", deq, "--ternary", "dequant"], check=True)
        ra, rb = gguf.GGUFReader(keep), gguf.GGUFReader(deq)
        ta = {x.name: x for x in ra.tensors}
        tb = {x.name: x for x in rb.tensors}
        qn = sorted(n for n in ta if n.endswith(".qweight"))
        assert len(qn) == 264, len(qn)
        assert bool(ra.fields["parakeet.ternary.present"].contents())
        assert int(ra.fields["parakeet.ternary.group_size"].contents()) == 128
        assert bool(ra.fields["parakeet.vad.present"].contents())
        for n in ("vad_head.proj.weight", "vad_head.ctx.weight", "vad_head.out.weight",
                  "vad_head.proj.bias", "vad_head.ctx.bias", "vad_head.out.bias"):
            assert n in ta and n in tb, n
        for n in qn[:11] + qn[-11:]:
            base = n[: -len(".qweight")]
            assert ta[n].tensor_type.name == "I8", ta[n].tensor_type
            assert (base + ".weight") not in ta
            qw = np.array(ta[n].data).view(np.uint8)
            sc = np.array(ta[base + ".scales"].data)
            want = np.array(tb[base + ".weight"].data)
            want = want.reshape(want.shape[0], -1)
            got = hfconv.unpack_ternary(qw, sc, want.shape[1], 128)
            assert np.array_equal(got, want), base
        print("check_hf_convert: keep-vs-dequant OK")


if __name__ == "__main__":
    for n in (1024, 4096, 130, 5, 128):
        test_unpack_matches_loop(n)
    test_unpack_rejects_bad_byte()
    test_packed_byte_validity()
    test_renames()
    print("check_hf_convert: OK")
    test_keep_matches_dequant()
