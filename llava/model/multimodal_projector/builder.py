import torch
import torch.nn as nn
import re


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads):
        super().__init__()

        self.pre_attn_norm = nn.LayerNorm(d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model)
        )

    def forward(self, x, y, mask=None):
        query_norm = self.pre_attn_norm(x)
        key_norm = self.pre_attn_norm(y)
        value_norm = self.pre_attn_norm(y)

        attn_out, _ = self.attn(query_norm, key_norm, value_norm, attn_mask=mask)
        attn_out = x + attn_out
        attn_out = self.post_attn_norm(attn_out)

        ff_out = self.ffn(attn_out)
        ff_out = attn_out + ff_out

        return ff_out


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class BimodalIterAttn(nn.Module):
    def __init__(self, d_vis, d_model, d_ffn, num_heads):
        super().__init__()

        self.text_xattn = CrossAttention(d_model, d_ffn, num_heads)

        self.vis_xattn = CrossAttention(d_model, d_ffn, num_heads)

    def vis_forward(self, x, y, mask=None):
        # x should be learned tokens
        return self.vis_xattn(x, y, mask)

    def text_forward(self, x, y, mask=None):
        # x should be learned tokens
        return self.text_xattn(x, y, mask)

    def forward(self, x, y, mask=None):
        raise NotImplementedError("Use vis_forward or text_forward")


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    # if projector_type == "bimodaliterattn":
    #     # Hardcoded for now
    #     return BimodalIterAttn(config.mm_hidden_size, config.hidden_size, config.hidden_size * 4, 8)

    raise ValueError(f"Unknown projector type: {projector_type}")
