import triton
import triton.language as tl

inv_ln2 = 1.44269504


@triton.jit
def fwd_decay_cumsum(
    g,
    g_o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    B,
    H,
    T,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DK: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    # output
    p_go = g_o + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(BT):
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += _g * inv_ln2
        tl.store(p_go, cum_decay.to(p_go.dtype.element_ty), mask=mask)
        p_g += DK
        p_go += DK


@triton.jit
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    s_qk_h: int,
    s_qk_t: int,
    s_qk_d: int,
    B: int,
    H: int,
    T: int,
    scale: float,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DK: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    # output
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < DK

    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK +
                         i_k * BK + tl.arange(0, BK))

    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(last_decay - _g)
        
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        
        p_q += DK
        p_g += DK
        p_k += DK
        p_kg += DK
        p_qg += DK
