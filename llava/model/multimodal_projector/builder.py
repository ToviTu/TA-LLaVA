from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import re
from copy import deepcopy
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2Attention,
        Gemma2FlashAttention2,
        rotate_half, 
        repeat_kv, 
        Gemma2MLP, 
        Gemma2RMSNorm
    )
from transformers.cache_utils import Cache

from transformers.utils import is_flash_attn_greater_or_equal, logging

logger = logging.get_logger(__name__)

def apply_rotary_pos_emb(h, cos, sin, position_ids=None, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    h_embed = (h * cos) + (rotate_half(h) * sin)
    return h_embed

def convert_attention_mask(query_len, attention_mask, dtype):
    # (batch_size, tgt_len) -> (batch_size, 1, query_len, tgt_len)
    min_dtype = torch.finfo(dtype).min
    mask = torch.full(attention_mask.size(), min_dtype).to(attention_mask.device)
    mask.masked_fill_(attention_mask==1, 0)
    mask = mask.unsqueeze(1).unsqueeze(1)
    return mask.repeat(1, 1, query_len, 1)


class Gemma2XAttention(Gemma2Attention):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
    
    def forward(
        self,
        src_hidden_states: torch.Tensor,
        tgt_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # KQV Projection
        src_bsz, src_q_len, _ = src_hidden_states.size()
        tgt_bsz, tgt_q_len, _ = tgt_hidden_states.size()

        query_states = self.q_proj(src_hidden_states)
        key_states = self.k_proj(tgt_hidden_states)
        value_states = self.v_proj(tgt_hidden_states)

        query_states = query_states.view(src_bsz, src_q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(tgt_bsz, tgt_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(tgt_bsz, tgt_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Positional encode the queries and keys
        src_position_ids = torch.arange(src_q_len, device=key_states.device).unsqueeze(0).repeat(src_bsz, 1)
        cos, sin = self.rotary_emb(query_states, src_position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)

        # Create tgt position ids from the attention mask
        tgt_position_ids = torch.arange(tgt_q_len, device=key_states.device).unsqueeze(0).repeat(tgt_bsz, 1)
        if attention_mask is not None:
            tgt_position_ids.masked_fill(attention_mask==0, -1)
        cos, sin = self.rotary_emb(key_states, tgt_position_ids)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        # Update cache
        # No cache since this thing is called once
        # if past_key_value is not None and use_cache:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {
        #         "sin": sin,
        #         "cos": cos,
        #         "sliding_window": self.sliding_window,
        #         "cache_position": cache_position,
        #     }
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Masking
        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping
        if attention_mask is not None:  # no matter the length, we just slice it
            mask = convert_attention_mask(src_q_len, attention_mask, attn_weights.dtype)
            mask = mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + mask

        # Attention output
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if tuple(attn_output.size()) != (src_bsz, self.num_heads, src_q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(src_bsz, self.num_heads, src_q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(src_bsz, src_q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Gemma2FlashXAttention(Gemma2FlashAttention2):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
    
    def forward(
        self,
        src_hidden_states: torch.Tensor,
        tgt_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # KQV Projection
        src_bsz, src_q_len, _ = src_hidden_states.size()
        tgt_bsz, tgt_q_len, _ = tgt_hidden_states.size()

        # I remember this wrong but it's actually the other way around
        query_states = self.q_proj(src_hidden_states)
        key_states = self.k_proj(tgt_hidden_states)
        value_states = self.v_proj(tgt_hidden_states)

        query_states = query_states.view(src_bsz, src_q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(tgt_bsz, tgt_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(tgt_bsz, tgt_q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Positional encode the queries and keys
        src_position_ids = torch.arange(src_q_len, device=key_states.device).unsqueeze(0).repeat(src_bsz, 1)
        cos, sin = self.rotary_emb(query_states, src_position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)

        # Create tgt position ids from the attention mask
        tgt_position_ids = torch.arange(tgt_q_len, device=key_states.device).unsqueeze(0).repeat(tgt_bsz, 1)
        if attention_mask is not None:
            tgt_position_ids.masked_fill(attention_mask==0, -1)
        cos, sin = self.rotary_emb(key_states, tgt_position_ids)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        # Update cache
        # if past_key_value is not None and use_cache:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {
        #         "sin": sin,
        #         "cos": cos,
        #         "sliding_window": self.sliding_window,
        #         "cache_position": cache_position,
        #     }
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if attention_mask is not None:
            seq_len = attention_mask.shape[1]
            key_states = key_states[:, :, :seq_len]
            value_states = value_states[:, :, :seq_len]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            src_q_len,
            dropout=dropout_rate,
            softmax_scale=self.scaling,
            is_causal=False, # not causal
            sliding_window=self.sliding_window,
            use_top_left_mask=False, # not causal
            softcap=self.config.attn_logit_softcapping if is_flash_attn_greater_or_equal("2.6.0") else None,
        )

        attn_output = attn_output.reshape(src_bsz, src_q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Gemma2EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2XAttention(config, layer_idx=layer_idx) \
            if config._attn_implementation != "flash_attention_2" \
            else Gemma2FlashXAttention(config, layer_idx=layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            src_hidden_states: torch.Tensor,
            tgt_hidden_states: torch.Tensor,
            tgt_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = src_hidden_states

        src_hidden_states = self.input_layernorm(src_hidden_states)
        tgt_hidden_states = self.input_layernorm(tgt_hidden_states)

        # Modified cross attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            src_hidden_states=src_hidden_states,
            tgt_hidden_states=tgt_hidden_states,
            attention_mask=tgt_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


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
    def __init__(self, config):
        super().__init__()

        self.text_xattn = Gemma2EncoderLayer(config, layer_idx=config.num_hidden_layers)

        self.vis_xattn = Gemma2EncoderLayer(config, layer_idx=config.num_hidden_layers+1)

        config.num_hidden_layers = config.num_hidden_layers + 2

    def vis_forward(self, **kwargs):
        # x should be learned tokens
        return self.vis_xattn(**kwargs)

    def text_forward(self, **kwargs):
        # x should be learned tokens
        return self.text_xattn(**kwargs)

    def forward(self, **kwargs):
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
