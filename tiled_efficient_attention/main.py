import torch

from model import GatedLinearAttention, GLAConfig

if __name__ == "__main__":

    batch, num_head, length, hidden = 32, 4, 2048, 2048

    config = GLAConfig(d_model=hidden, n_head=num_head)
    print(config)

    GLA = GatedLinearAttention(config,
                               mode="fused_chunk").cuda().to(torch.bfloat16)

    x = torch.randn((batch, length, hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                    requires_grad=False)

    y, _ = GLA(x)
    print(y.shape)
