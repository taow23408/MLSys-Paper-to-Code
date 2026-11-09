"""FlashAttention Forward 填空练习。

要求：仅填补 TODO_XX 标识符所处的表达式；不要改动测试和接口。
输入：连续 FP16 (B, H, N, D)，Self-Attention，支持 Causal 与非 Causal。
依赖：torch、triton；运行：python FlashAttention_填空练习.py
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_ptr, V_ptr,
    off_hz, offs_m,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    qk_scale: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    kv_base = off_hz * N_CTX * HEAD_DIM
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in range(0, N_CTX, BLOCK_N):
        kv_rows = start_n + offs_n
        kv_offsets = TODO_03_KV_OFFSETS
        kv_mask = TODO_04_KV_MASK

        k = tl.load(K_ptr + kv_offsets, mask=kv_mask, other=0.0)
        v = tl.load(V_ptr + kv_offsets, mask=kv_mask, other=0.0)

        qk = TODO_05_QK_SCORE

        if CAUSAL:
            valid = TODO_06_CAUSAL_VALID
        else:
            valid = kv_rows[None, :] < N_CTX
        qk = tl.where(valid, qk, float('-inf'))

        m_ij = TODO_07_NEW_MAX
        p = TODO_08_EXP2_PROB
        alpha = TODO_09_ALPHA
        l_ij = tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        acc = TODO_10_ACC_DOT
        l_i = TODO_11_L_UPDATE
        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N_CTX,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    qk_scale: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = off_hz * N_CTX * HEAD_DIM
    q_offsets = TODO_01_Q_OFFSETS
    q_mask = TODO_02_Q_MASK
    q = tl.load(Q_ptr + q_offsets, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_ptr, V_ptr,
        off_hz, offs_m, N_CTX,
        HEAD_DIM, BLOCK_N, qk_scale, CAUSAL,
    )

    output = TODO_12_NORMALIZE
    tl.store(O_ptr + q_offsets, output, mask=q_mask)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError('Q/K/V 必须在 CUDA GPU 上')
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError('Q/K/V 必须是相同形状的 (B, H, N, D)')
    if not (q.dtype == k.dtype == v.dtype == torch.float16):
        raise ValueError('教学版本仅支持 FP16 输入')
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError('教学版本要求 Q/K/V 连续存储')

    Z, H, N_CTX, HEAD_DIM = q.shape
    if N_CTX < 1 or HEAD_DIM not in (32, 64, 128):
        raise ValueError('需要 N>=1 且 D 为 32、64 或 128')

    BLOCK_M = 16
    BLOCK_N = 32
    qk_scale = math.log2(math.e) / math.sqrt(HEAD_DIM)
    grid = TODO_13_GRID
    out = torch.empty_like(q)

    _attn_fwd[grid](
        q, k, v, out,
        N_CTX,
        H=H, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        qk_scale=qk_scale,
        CAUSAL=causal,
        num_warps=4,
    )
    return out


def attention_reference(q, k, v, causal=True):
    score = q.float() @ k.float().transpose(-2, -1)
    score = score / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        future = torch.triu(torch.ones((n, n), device=q.device, dtype=torch.bool), diagonal=1)
        score = score.masked_fill(future, float('-inf'))
    probability = torch.softmax(score, dim=-1)
    return (probability @ v.float()).to(q.dtype)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('需要 CUDA GPU')
    torch.manual_seed(42)

    cases = [(1, 1, 12, 32), (2, 4, 128, 64), (2, 4, 130, 64),
             (1, 8, 257, 32), (1, 2, 64, 128)]
    for causal in (False, True):
        for z, h, n, d in cases:
            q, k, v = [torch.randn((z, h, n, d), device='cuda', dtype=torch.float16)
                       for _ in range(3)]
            got = flash_attention(q, k, v, causal=causal)
            ref = attention_reference(q, k, v, causal=causal)
            max_error = (got.float() - ref.float()).abs().max().item()
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
            print(f'causal={str(causal):5s} shape={(z,h,n,d)} '
                  f'max_abs_error={max_error:.3e} PASS')
