from typing import Optional, Tuple

from .utils import contiguous

import torch
import triton

import triton.language as tl


@triton.autotune(configs=[
    triton.Config({'BS': 16}, num_warps=2),
    triton.Config({'BS': 16}, num_warps=4),
    triton.Config({'BS': 16}, num_warps=8),
    triton.Config({'BS': 32}, num_warps=2),
    triton.Config({'BS': 32}, num_warps=4),
    triton.Config({'BS': 32}, num_warps=8),
    triton.Config({'BS': 64}, num_warps=2),
    triton.Config({'BS': 64}, num_warps=4),
    triton.Config({'BS': 64}, num_warps=8),
],
                 key=['S'])
@triton.jit
def chunk_gla_fwd_kernel_cum(s, o, s_s_h, s_s_t, s_s_d, T: tl.constexpr,
                             S: tl.constexpr, BT: tl.constexpr,
                             BS: tl.constexpr):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d),
                            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d),
                            (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gla_fwd_kernel_h(k, v, g, h, h0, ht, s_k_h, s_k_t, s_k_d, s_v_h,
                           s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, T: tl.constexpr,
                           K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                           BK: tl.constexpr, BV: tl.constexpr,
                           NT: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
                           STORE_FINAL_STATE: tl.constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1),
                                (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
                                (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                                (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V),
                                (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV),
                                (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
                                (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K, ), (s_k_d, ),
                                 ((i_t * BT + BT - 1) * K + i_k * BK, ),
                                 (BK, ), (0, ))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BT]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        if i_t < NT - 1:
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0, ))
        else:
            b_gn = tl.min(b_g, axis=1)
        b_h *= tl.exp(b_gn)[:, None]
        b_k = (b_k * tl.exp(b_gn[:, None] - b_g)).to(b_k.dtype)
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1),
                                (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gla_fwd_kernel_intra(q, k, g, A, s_k_h, s_k_t, s_k_d, scale,
                               T: tl.constexpr, K: tl.constexpr,
                               BT: tl.constexpr, BC: tl.constexpr,
                               BK: tl.constexpr, NC: tl.constexpr):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c %
                                                                (NC * NC)) % NC
    n_bh = tl.num_programs(2)

    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
                                (i_k * BK, i_t * BT + i_j * BC), (BK, BC),
                                (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t),
                                 (i_k * BK, i_t * BT + i_j * BC), (BK, BC),
                                 (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K, ), (s_k_d, ),
                                 ((i_t * BT + i_i * BC) * K + i_k * BK, ),
                                 (BK, ), (0, ))
        p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT),
                                (BT, 1), (i_t * BT + i_i * BC, i_j * BC),
                                (BC, BC), (1, 0))
        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0, ))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g - b_gn[None, :]) * scale).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[:, None] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_qg, b_kg, allow_tf32=False)
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT + i_i * BC, i_k * BK), (BC, BK),
                                (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K, ), (s_k_d, ),
                                ((i_t * BT + i_j * BC) * K + i_k * BK, ),
                                (BK, ), (0, ))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T * K, ), (s_k_d, ),
                                 ((i_t * BT + i_j * BC) * K + i_k * BK, ),
                                 (BK, ), (0, ))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))

        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC +
                                              tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        for j in range(0, BC):
            # [BK,]
            b_k = tl.load(p_k, boundary_check=(0, )).to(tl.float32)
            b_gk = tl.load(p_gk, boundary_check=(0, )).to(tl.float32)
            # [BC,]
            b_A = tl.sum(
                b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) * scale, 1)
            b_A = tl.where(o_i >= j, b_A, 0.)
            tl.store(A + o_A + j, b_A.to(b_q.dtype), mask=m_A)

            p_k = tl.advance(p_k, (K, ))
            p_gk = tl.advance(p_gk, (K, ))


@triton.jit
def chunk_gla_fwd_kernel_inter(q, v, g, h, o, A, s_k_h, s_k_t, s_k_d, s_v_h,
                               s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale,
                               T: tl.constexpr, K: tl.constexpr,
                               V: tl.constexpr, BT: tl.constexpr,
                               BK: tl.constexpr, BV: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                                (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V),
                                (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV),
                                (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0),
                            (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state,
                checkpoint_level):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        NV = triton.cdiv(V, BV)
        num_warps = 4 if BK == 64 else 2
        num_stages = 1

        def fwd_inner(q,
                      k,
                      v,
                      g,
                      B,
                      H,
                      T,
                      K,
                      V,
                      BT,
                      BK,
                      BV,
                      NT,
                      h0=None,
                      ht=None):
            NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
            h = q.new_empty(B, H, NT * K, V)
            grid = (NV, NK, B * H)
            chunk_gla_fwd_kernel_h[grid](k,
                                         v,
                                         g,
                                         h,
                                         h0,
                                         ht,
                                         k.stride(1),
                                         k.stride(2),
                                         k.stride(3),
                                         v.stride(1),
                                         v.stride(2),
                                         v.stride(3),
                                         h.stride(1),
                                         h.stride(2),
                                         h.stride(3),
                                         T=T,
                                         K=K,
                                         V=V,
                                         BT=BT,
                                         BK=BK,
                                         BV=BV,
                                         NT=NT,
                                         USE_INITIAL_STATE=h0 is not None,
                                         STORE_FINAL_STATE=ht is not None,
                                         num_warps=num_warps,
                                         num_stages=num_stages)
            return h

        final_state = None
        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float)

        g_org, g = g, torch.empty_like(g, dtype=torch.float)

        def grid(meta):
            return ((triton.cdiv(meta['S'], meta['BS']), NT, B * H))

        # keep cummulative normalizer in fp32
        # this kernel is equivalent to
        # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
        chunk_gla_fwd_kernel_cum[grid](g_org,
                                       g,
                                       g.stride(1),
                                       g.stride(2),
                                       g.stride(3),
                                       T=T,
                                       S=K,
                                       BT=BT)
        h = fwd_inner(q=q,
                      k=k,
                      v=v,
                      g=g,
                      B=B,
                      H=H,
                      T=T,
                      K=K,
                      V=V,
                      BT=BT,
                      BK=BK,
                      BV=BV,
                      NT=NT,
                      h0=initial_state if initial_state is not None else None,
                      ht=final_state if final_state is not None else None)
        A = q.new_zeros(NK, B, H, T, BT)
        grid = (NK, NT * NC * NC, B * H)
        chunk_gla_fwd_kernel_intra[grid](q,
                                         k,
                                         g,
                                         A,
                                         k.stride(1),
                                         k.stride(2),
                                         k.stride(3),
                                         scale,
                                         T=T,
                                         K=K,
                                         BT=BT,
                                         BC=BC,
                                         BK=BK,
                                         NC=NC,
                                         num_warps=num_warps,
                                         num_stages=num_stages)
        A = A.sum(0, dtype=A.dtype)
        o = torch.empty_like(v)
        grid = (NV, NT, B * H)
        chunk_gla_fwd_kernel_inter[grid](q,
                                         v,
                                         g,
                                         h,
                                         o,
                                         A,
                                         k.stride(1),
                                         k.stride(2),
                                         k.stride(3),
                                         v.stride(1),
                                         v.stride(2),
                                         v.stride(3),
                                         h.stride(1),
                                         h.stride(2),
                                         h.stride(3),
                                         scale,
                                         T=T,
                                         K=K,
                                         V=V,
                                         BT=BT,
                                         BK=BK,
                                         BV=BV,
                                         num_warps=num_warps,
                                         num_stages=num_stages)
        if checkpoint_level >= 1:
            del g
            g = g_org
        if checkpoint_level > 1:
            del h
            h, initial_state = None, None

        ctx.save_for_backward(q, k, v, g, h, initial_state, A)
        ctx.BT = BT
        ctx.scale = scale
        ctx.checkpoint_level = checkpoint_level
        return o, final_state


def chunk_gla(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        scale: Optional[int] = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        checkpoint_level: Optional[int] = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, H, T, K)`
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        g (torch.Tensor):
            Forget gates of shape `(B, H, T, K)` applied to keys.
        scale (Optional[int]):
            Scale factor for the GLA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `0`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
    """
    assert checkpoint_level in [0, 1, 2]
    if scale is None:
        scale = q.shape[-1]**-0.5
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state,
                                            output_final_state,
                                            checkpoint_level)
    return o, final_state
