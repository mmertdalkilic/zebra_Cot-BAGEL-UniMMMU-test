# -*- coding: utf-8 -*-
"""Fix AutoAWQ Triton unpack kernels on Blackwell / new Triton.

Stock kernels do `iweights >> shifts` after `tl.arange` / `tl.load(..., other=0.0)`.
Recent Triton types those as fp16/fp32, which raises:

    IncompatibleTypeErrorImpl('invalid operands of type triton.language.float16 ...')

We re-register the kernels with explicit int32 unpack, smoke-test a tiny dequant,
and if that still fails set TRITON_AVAILABLE=False so AutoAWQ uses its PyTorch
dequant + matmul path (correct, slower).

Set AWQ_FORCE_PYTORCH_DEQUANT=1 to skip Triton entirely.
"""
from __future__ import annotations

import os
import sys


_PACKED = ("qweight", "qzeros")


def _install_accelerate_int_hooks() -> None:
    """Stop device_map + torch_dtype from casting packed AWQ ints to Half."""
    try:
        import accelerate.utils.modeling as acc_modeling
    except Exception as e:
        print(f"[awq] accelerate hook skipped: {e}", flush=True)
        return
    orig = acc_modeling.set_module_tensor_to_device
    if getattr(orig, "_awq_int_guard", False):
        return

    def wrapped(module, tensor_name, device, value=None, dtype=None, **kwargs):
        short = str(tensor_name).rsplit(".", 1)[-1]
        if short in _PACKED:
            dtype = None
        return orig(module, tensor_name, device, value=value, dtype=dtype, **kwargs)

    wrapped._awq_int_guard = True
    acc_modeling.set_module_tensor_to_device = wrapped
    print("[awq] accelerate hook: never cast qweight/qzeros", flush=True)


def _patch_unpack_awq() -> None:
    try:
        import awq.utils.packing_utils as pu
    except Exception:
        return
    orig = pu.unpack_awq
    if getattr(orig, "_int_guard", False):
        return

    def unpack_awq(qweight, qzeros, bits):
        if qweight is not None and qweight.is_floating_point():
            raise TypeError(
                f"AWQ qweight is {qweight.dtype}, expected int32. "
                "Do not pass torch_dtype=float16 into from_pretrained for AWQ models."
            )
        if qzeros is not None and qzeros.is_floating_point():
            raise TypeError(f"AWQ qzeros is {qzeros.dtype}, expected int32")
        return orig(qweight, qzeros, bits)

    unpack_awq._int_guard = True
    pu.unpack_awq = unpack_awq


def prepare_awq_model(model, label: str = "model"):
    """Verify packed weights are int32; cast only floating tensors to fp16."""
    import torch

    sample = None
    bad = []
    n_packed = 0
    for name, buf in model.named_buffers():
        short = name.rsplit(".", 1)[-1]
        if short not in _PACKED:
            continue
        n_packed += 1
        if sample is None and short == "qweight":
            sample = (name, str(buf.dtype), tuple(buf.shape))
        if buf.dtype not in (
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.uint32,
        ):
            bad.append((name, str(buf.dtype)))
    if sample:
        print(f"[{label}] qweight {sample[0]} dtype={sample[1]} shape={sample[2]}", flush=True)
    if n_packed == 0:
        print(f"[{label}] no AWQ qweight buffers found", flush=True)
        return
    if bad:
        raise TypeError(
            f"[{label}] {len(bad)} packed AWQ tensors are float, e.g. {bad[0]}. "
            "Load without torch_dtype so qweight stays int32."
        )
    n_cast = 0
    dtype = torch.float16
    for child in model.modules():
        for _, p in child.named_parameters(recurse=False):
            if p is not None and p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype)
                n_cast += 1
        for bname, b in list(child.named_buffers(recurse=False)):
            if b is None or bname in _PACKED:
                continue
            if b.is_floating_point() and b.dtype != dtype:
                child._buffers[bname] = b.to(dtype)
                n_cast += 1
    print(f"[{label}] packed int32 ok; cast {n_cast} float tensors to fp16", flush=True)


def _force_pytorch_dequant(reason: str) -> None:
    try:
        import awq.modules.linear.gemm as gemm

        gemm.TRITON_AVAILABLE = False
        print(f"[awq] Triton disabled -> PyTorch dequant ({reason})", flush=True)
    except Exception as e:
        print(f"[awq] could not disable Triton: {e}", flush=True)


def apply_awq_triton_patch() -> None:
    _install_accelerate_int_hooks()
    _patch_unpack_awq()
    if os.environ.get("AWQ_FORCE_PYTORCH_DEQUANT", "").strip() in {"1", "true", "True"}:
        _force_pytorch_dequant("AWQ_FORCE_PYTORCH_DEQUANT=1")
        return

    try:
        import torch
        import triton
        import triton.language as tl
        import awq.modules.triton.gemm as awq_triton
    except Exception as e:
        _force_pytorch_dequant(f"import failed: {e}")
        return

    @triton.jit
    def awq_dequantize_kernel_i32(
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        group_size,
        result_ptr,
        num_cols,
        num_rows,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
    ):
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)

        offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y, dtype=tl.int32)
        offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X, dtype=tl.int32)
        offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

        masks_y = offsets_y < num_rows
        masks_x = offsets_x < num_cols
        masks = masks_y[:, None] & masks_x[None, :]

        result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y, dtype=tl.int32)
        result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
            0, BLOCK_SIZE_X * 8, dtype=tl.int32
        )
        result_offsets = (
            8 * num_cols * result_offsets_y[:, None] + result_offsets_x[None, :]
        )

        result_masks_y = result_offsets_y < num_rows
        result_masks_x = result_offsets_x < num_cols * 8
        result_masks = result_masks_y[:, None] & result_masks_x[None, :]

        iweights = tl.load(qweight_ptr + offsets, masks, other=0).to(tl.int32)
        iweights = tl.interleave(iweights, iweights)
        iweights = tl.interleave(iweights, iweights)
        iweights = tl.interleave(iweights, iweights)
        iweights = iweights.to(tl.int32)

        reverse_awq_order_tensor = (
            (tl.arange(0, 2, dtype=tl.int32) * 4)[None, :]
            + tl.arange(0, 4, dtype=tl.int32)[:, None]
        ).reshape(8)

        shifts = (reverse_awq_order_tensor * 4).to(tl.int32)
        shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
        shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

        iweights = (iweights >> shifts) & 0xF

        zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(
            0, 1, dtype=tl.int32
        )
        zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X, dtype=tl.int32)
        zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

        zero_masks_y = zero_offsets_y < num_rows // group_size
        zero_masks_x = zero_offsets_x < num_cols
        zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

        zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, other=0).to(tl.int32)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8)).to(tl.int32)

        zeros = (zeros >> shifts) & 0xF

        scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(
            0, 1, dtype=tl.int32
        )
        scale_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
            0, BLOCK_SIZE_X * 8, dtype=tl.int32
        )
        scale_offsets = (
            num_cols * 8 * scale_offsets_y[:, None] + scale_offsets_x[None, :]
        )
        scale_masks_y = scale_offsets_y < num_rows // group_size
        scale_masks_x = scale_offsets_x < num_cols * 8
        scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

        scales = tl.load(scales_ptr + scale_offsets, scale_masks)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

        iweights = (iweights - zeros) * scales
        iweights = iweights.to(result_ptr.type.element_ty)
        tl.store(result_ptr + result_offsets, iweights, result_masks)

    @triton.jit
    def awq_gemm_kernel_i32(
        a_ptr,
        b_ptr,
        c_ptr,
        zeros_ptr,
        scales_ptr,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        pid_z = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        accumulator_dtype = c_ptr.type.element_ty
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

        reverse_awq_order_tensor = (
            (tl.arange(0, 2, dtype=tl.int32) * 4)[None, :]
            + tl.arange(0, 4, dtype=tl.int32)[:, None]
        ).reshape(8)

        shifts = (reverse_awq_order_tensor * 4).to(tl.int32)
        shifts = tl.broadcast_to(
            shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8)
        )
        shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M, dtype=tl.int32)
        masks_am = offsets_am < M

        offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(
            0, BLOCK_SIZE_N // 8, dtype=tl.int32
        )
        masks_bn = offsets_bn < N // 8

        offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(
            0, BLOCK_SIZE_N // 8, dtype=tl.int32
        )
        masks_zn = offsets_zn < N // 8

        offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N, dtype=tl.int32)
        masks_sn = offsets_sn < N

        offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K, dtype=tl.int32)
        offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
        offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

        a_ptrs = a_ptr + offsets_a
        b_ptrs = b_ptr + offsets_b

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
            masks_k = offsets_k < K
            masks_a = masks_am[:, None] & masks_k[None, :]
            a = tl.load(a_ptrs, mask=masks_a)

            masks_b = masks_k[:, None] & masks_bn[None, :]
            b = tl.load(b_ptrs, mask=masks_b, other=0).to(tl.int32)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = b.to(tl.int32)

            offsets_szk = (
                BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K
            ) // group_size + tl.arange(0, 1, dtype=tl.int32)
            offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
            masks_zk = offsets_szk < K // group_size
            masks_z = masks_zk[:, None] & masks_zn[None, :]
            zeros = tl.load(zeros_ptr + offsets_z, mask=masks_z, other=0).to(tl.int32)
            zeros = tl.interleave(zeros, zeros)
            zeros = tl.interleave(zeros, zeros)
            zeros = tl.interleave(zeros, zeros)
            zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N)).to(tl.int32)

            offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
            masks_sk = offsets_szk < K // group_size
            masks_s = masks_sk[:, None] & masks_sn[None, :]
            scales = tl.load(scales_ptr + offsets_s, mask=masks_s)
            scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

            b = (b >> shifts) & 0xF
            zeros = (zeros >> shifts) & 0xF
            b = (b - zeros) * scales
            b = b.to(c_ptr.type.element_ty)

            accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

            offsets_k += BLOCK_SIZE_K * SPLIT_K
            a_ptrs += BLOCK_SIZE_K * SPLIT_K
            b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M, dtype=tl.int32)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N, dtype=tl.int32)
        c_ptrs = c_ptr + N * offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if SPLIT_K == 1:
            tl.store(c_ptrs, c, mask=c_mask)
        else:
            tl.atomic_add(c_ptrs, c, mask=c_mask)

    awq_triton.awq_dequantize_kernel = awq_dequantize_kernel_i32
    awq_triton.awq_gemm_kernel = awq_gemm_kernel_i32
    print("[awq] installed int32 Triton unpack kernels", flush=True)

    if not torch.cuda.is_available():
        print("[awq] no CUDA — skipped kernel smoke test", flush=True)
        return

    try:
        qweight = torch.zeros(128, 16, dtype=torch.int32, device="cuda")
        scales = torch.ones(1, 128, dtype=torch.float16, device="cuda")
        zeros = torch.zeros(1, 16, dtype=torch.int32, device="cuda")
        out = awq_triton.awq_dequantize_triton(qweight, scales, zeros)
        x = torch.zeros(1, 128, dtype=torch.float16, device="cuda")
        _ = awq_triton.awq_gemm_triton(x, qweight, scales, zeros, split_k_iters=1)
        print(f"[awq] Triton smoke test OK  dequant={tuple(out.shape)}", flush=True)
    except Exception as e:
        err = str(e).split("\n")[0][:240]
        print(f"[awq] Triton smoke test failed: {err}", flush=True)
        _force_pytorch_dequant("smoke test failed")


if __name__ == "__main__":
    apply_awq_triton_patch()
    sys.exit(0)
