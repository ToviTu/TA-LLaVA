from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2Model

from ..tallava_arch import TALlavaMetaModel, TALlavaMetaForCausalLM
from transformers.generation.utils import GenerateOutput


from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)


class TALlavaConfig(Gemma2Config):
    model_type = "tallava_gemma"


class TALlavaGemmaModel(TALlavaMetaModel, Gemma2Model):
    config_class = TALlavaConfig

    def __init__(self, config: Gemma2Config):
        super(TALlavaGemmaModel, self).__init__(config)


class TALlavaGemmaForCausalLM(Gemma2ForCausalLM, TALlavaMetaForCausalLM):
    config_class = TALlavaConfig

    def __init__(self, config):
        super(Gemma2ForCausalLM, self).__init__(config)
        self.model = TALlavaGemmaModel(config)
        self.pretraining_tp = 1
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if kwargs["input_ids"].shape[1] == 1:
            return self.text_forward(**kwargs)

        if "images" in kwargs and kwargs["images"] is not None:
            return self.vis_forward(**kwargs)
        else:
            return self.text_forward(**kwargs)

    def text_forward(
        self,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # assert "images" not in kwargs, "images should not be in kwargs"
        return super().forward(**kwargs)

    def vis_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # 2D or 4D?
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        num_logits_to_keep = (
            input_ids.shape[1] if num_logits_to_keep == 0 else num_logits_to_keep
        )
        assert images is not None, "images should not be None"
        assert inputs_embeds is None, "inputs_embeds not supported"

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.get_model().gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # Create the instruction mask based on the input ids
        instruction_mask = torch.zeros_like(input_ids)
        assert torch.all(
            torch.any(input_ids == 2516, dim=1), dim=0
        ), "'model' token not found in input ids"
        for i in range(input_ids.shape[0]):
            instruction_mask[i, : torch.argmax((input_ids[i] == 2516).float())] = 1

        # Adjust attention mask length
        # Currently assuming 2D attention mask
        attention_mask_ = torch.ones(
            input_ids.shape[0],
            self.config.num_learnable_tokens + input_ids.shape[1],
        ).to(input_ids.device)
        if attention_mask is not None:
            for i, mask_ in enumerate(attention_mask_):  # Copy the mask
                mask_[self.config.num_learnable_tokens :] = attention_mask[i]
        attention_mask = attention_mask_

        # Forward pass
        vis_priori = (
            self.get_model()
            .vision_priori(
                torch.arange(self.config.num_learnable_tokens).to(input_ids.device)
            )
            .unsqueeze(0)
            .repeat(input_ids.size(0), 1, 1)
        )
        image_embeds = self.encode_images(images).to(dtype=self.dtype)
        text_embeds = self.get_model().embed_tokens(input_ids)

        # Concat the visual priori with the text embeddings
        inputs_embeds = torch.cat((vis_priori, text_embeds), dim=1)

        # Cache: taken from the original gemma2 implementation
        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        # Work around: what does this mean???
        if past_key_values is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        else:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        position_ids = cache_position.unsqueeze(0)

        causal_mask = self.get_model()._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        # Decoding
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.get_model().layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Visual feature pooling
            if i < len(self.get_model().layers) - 1:
                mem_embeds = hidden_states[:, : self.config.num_learnable_tokens]
                text_embeds = hidden_states[:, self.config.num_learnable_tokens :]

                if i % 4 == 0:
                    layer_outputs = self.get_model().bottle_neck.text_forward(
                        src_hidden_states=mem_embeds,
                        tgt_hidden_states=text_embeds,
                        tgt_attention_mask=instruction_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    mem_embeds = layer_outputs[0]
                if i % 4 == 1:
                    layer_outputs = self.get_model().bottle_neck.vis_forward(
                        src_hidden_states=mem_embeds,
                        tgt_hidden_states=image_embeds,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    mem_embeds = layer_outputs[0]

                hidden_states = torch.cat((mem_embeds, text_embeds), dim=1)

        hidden_states = self.get_model().norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        outputs = tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=all_self_attns,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        pixel_values=None,
        images=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            pixel_values=pixel_values,
            **kwargs,
        )
        # Dont pass attention mask to the model
        inputs["attention_mask"] = None

        # Recalculate the cache position if there are images
        if past_key_values is not None and images is not None:
            if past_key_values.get_seq_length() == 0:
                inputs["cache_position"] = torch.arange(
                    self.config.num_learnable_tokens + input_ids.shape[1],
                    device=input_ids.device,
                )
            else:
                inputs["cache_position"] = torch.tensor(
                    [past_key_values.get_seq_length()], device=input_ids.device
                )

        inputs["position_ids"] = inputs["cache_position"].unsqueeze(0)

        if input_ids.shape[1] == 1:
            return inputs

        if images is not None:
            inputs["images"] = images
        elif pixel_values is not None:
            inputs["images"] = pixel_values

        if inputs["position_ids"].shape == 1:
            inputs["images"] = None

        return inputs


AutoConfig.register("tallava_gemma", TALlavaConfig)
AutoModelForCausalLM.register(TALlavaConfig, TALlavaGemmaForCausalLM)
