from typing import Optional, Tuple, Union

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .chunk import chunk_gla
from .chunk_fuse import fused_chunk_gla
from .recurrent_fuse import fused_recurrent_gla


class GatedLinearAttention(nn.Module):

    def __init__(self, config, mode='chunk'):
        super().__init__()

        self.mode = mode
        self.embed_dim = config.d_model
        self.num_heads = config.n_head

        self.gate_fn = nn.functional.silu
        assert config.use_gk and not config.use_gv, "Only use_gk is supported for simplicity."

        self.q_proj = nn.Linear(self.embed_dim,
                                self.embed_dim // 2,
                                bias=False)
        self.k_proj = nn.Linear(self.embed_dim,
                                self.embed_dim // 2,
                                bias=False)
        self.k_gate = nn.Sequential(nn.Linear(self.embed_dim, 16, bias=False),
                                    nn.Linear(16, self.embed_dim // 2))

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.head_dim = self.embed_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5
        self.group_norm = nn.LayerNorm(self.head_dim,
                                       eps=1e-5,
                                       elementwise_affine=False)

    def forward(self, x, hidden_states=None):
        # x has shape [batch, seq_len, hidden_dim]
        q = self.q_proj(x)
        k = self.k_proj(x) * self.scaling
        k_gate = self.k_gate(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        output, new_hidden_states = self.gated_linear_attention(
            q, k, v, k_gate, hidden_states=hidden_states)

        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output, new_hidden_states

    def gated_linear_attention(self,
                               q,
                               k,
                               v,
                               gk,
                               normalizer=16,
                               hidden_states=None):
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads).contiguous()
        gk = rearrange(gk, 'b l (h d) -> b h l d',
                       h=self.num_heads).contiguous()
        gk = F.logsigmoid(gk) / normalizer

        if self.mode == 'fused_chunk':
            o, new_hidden_states = fused_chunk_gla(q,
                                                   k,
                                                   v,
                                                   gk,
                                                   initial_state=hidden_states,
                                                   output_final_state=True)
        elif self.mode == 'fused_recurrent':
            o = fused_recurrent_gla(q, k, v, gk)
            new_hidden_states = None
        elif self.mode == 'chunk':
            o, new_hidden_states = chunk_gla(q,
                                             k,
                                             v,
                                             gk,
                                             initial_state=hidden_states,
                                             output_final_state=True)
        o = self.group_norm(o)
        o = rearrange(o, 'b h l d -> b l (h d)')
        return o, new_hidden_states
