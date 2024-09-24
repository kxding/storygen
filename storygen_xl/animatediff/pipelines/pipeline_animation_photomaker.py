# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

import PIL

# from collections import OrderedDict
import numpy as np

from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor
from dataclasses import dataclass

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from animatediff.models.unet import UNet3DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    BaseOutput,
)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
try:
    from diffusers.pipeline_utils import DiffusionPipeline
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.utils import (
    _get_model_file,
    is_transformers_available,
    logging,
)
from utils import PhotoMakerIDEncoder

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


@dataclass
class AnimatePipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    videos: Union[torch.Tensor, np.ndarray]



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class AnimationPipeline(DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    # def encode_prompt_with_photomakerTriggerWord(
    #     self,
    #     prompt: str,
    #     prompt_2: Optional[str] = None,
    #     device: Optional[torch.device] = None,
    #     num_videos_per_prompt: int = 1,
    #     do_classifier_free_guidance: bool = True,
    #     negative_prompt: Optional[str] = None,
    #     negative_prompt_2: Optional[str] = None,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     lora_scale: Optional[float] = None,
    #     class_tokens_mask: Optional[torch.LongTensor] = None,
    #     nc_flag = False,
    #     num_id_images: int = 1,
    # ):
    #     r"""
    #     Encodes the prompt into text encoder hidden states.

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             prompt to be encoded
    #         prompt_2 (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
    #             used in both text-encoders
    #         device: (`torch.device`):
    #             torch device
    #         num_videos_per_prompt (`int`):
    #             number of images that should be generated per prompt
    #         do_classifier_free_guidance (`bool`):
    #             whether to use classifier free guidance or not
    #         negative_prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts not to guide the image generation. If not defined, one has to pass
    #             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
    #             less than `1`).
    #         negative_prompt_2 (`str` or `List[str]`, *optional*):
    #             The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
    #             `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
    #         prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
    #             provided, text embeddings will be generated from `prompt` input argument.
    #         negative_prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
    #             weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
    #             argument.
    #         pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
    #             If not provided, pooled text embeddings will be generated from `prompt` input argument.
    #         negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
    #             weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
    #             input argument.
    #         lora_scale (`float`, *optional*):
    #             A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
    #     """
    #     device = device or self._execution_device

    #     # set lora scale so that monkey patched LoRA
    #     # function of text encoder can correctly access it
    #     if lora_scale is not None and isinstance(self, LoraLoaderMixin):
    #         self._lora_scale = lora_scale

    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]
            
    #     #Find the token id of the trigger word
    #     image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

    #     # Define tokenizers and text encoders
    #     tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
    #     text_encoders = (
    #         [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
    #     )

    #     if prompt_embeds is None:
    #         prompt_2 = prompt_2 or prompt
    #         # textual inversion: procecss multi-vector tokens if necessary
    #         prompt_embeds_list = []
    #         prompts = [prompt, prompt_2]
    #         for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

    #             if isinstance(self, TextualInversionLoaderMixin):
    #                 prompt = self.maybe_convert_prompt(prompt, tokenizer)
                
    #             input_ids = tokenizer.encode(prompt)
    #             clean_index = 0
    #             clean_input_ids = []
    #             class_token_index = []
                
    #             for i, token_id in enumerate(input_ids):
    #                 if token_id == image_token_id:
    #                     class_token_index.append(clean_index - 1)
    #                 else:
    #                     clean_input_ids.append(token_id)
    #                     clean_index += 1
    #             if nc_flag:
    #                 return None, None, None
    #             if len(class_token_index) > 1:
    #                 raise ValueError(
    #                     f"Multiple class tokens found in the prompt. Please make sure that the prompt contains only one class token. \
    #                     Trigger word: {self.trigger_word}, Prompt: {prompt}"
    #                 )
                    
    #             class_token_index = class_token_index[0]
                
    #             # Expand the class word token and corresponding mask
    #             class_token = clean_input_ids[class_token_index]
    #             clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + clean_input_ids[class_token_index + 1:]
                 
    #             # Truncation or Padding
    #             max_len = tokenizer.model_max_length
    #             if len(clean_input_ids) > max_len:
    #                 clean_input_ids = clean_input_ids[:max_len]
    #             else:
    #                 clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (max_len - len(clean_input_ids))
                
    #             class_tokens_mask = [True if class_token_index <= i < class_token_index + num_id_images else False \
    #                 for i in range(len(clean_input_ids))]
                
    #             text_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
    #             class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.long).unsqueeze(0)

    #             text_inputs = tokenizer(
    #                 prompt,
    #                 padding="max_length",
    #                 max_length=tokenizer.model_max_length,
    #                 truncation=True,
    #                 return_tensors="pt",
    #             )

    #             # text_input_ids = text_inputs.input_ids  #! represented by text_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
    #             untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    #             if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
    #                 text_input_ids, untruncated_ids
    #             ):
    #                 removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
    #                 logger.warning(
    #                     "The following part of your input was truncated because CLIP can only handle sequences up to"
    #                     f" {tokenizer.model_max_length} tokens: {removed_text}"
    #                 )

    #             prompt_embeds = text_encoder(
    #                 text_input_ids.to(device),
    #                 output_hidden_states=True,
    #             )

    #             # We are only ALWAYS interested in the pooled output of the final text encoder
    #             pooled_prompt_embeds = prompt_embeds[0]
    #             prompt_embeds = prompt_embeds.hidden_states[-2]

    #             prompt_embeds_list.append(prompt_embeds)

    #         prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    #     # get unconditional embeddings for classifier free guidance
    #     zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
    #     if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
    #         negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    #         negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    #     elif do_classifier_free_guidance and negative_prompt_embeds is None:
    #         negative_prompt = negative_prompt or ""
    #         negative_prompt_2 = negative_prompt_2 or negative_prompt

    #         uncond_tokens: List[str]
    #         if prompt is not None and type(prompt) is not type(negative_prompt):
    #             raise TypeError(
    #                 f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
    #                 f" {type(prompt)}."
    #             )
    #         elif isinstance(negative_prompt, str):
    #             uncond_tokens = [negative_prompt, negative_prompt_2]
    #         elif batch_size != len(negative_prompt):
    #             raise ValueError(
    #                 f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
    #                 f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
    #                 " the batch size of `prompt`."
    #             )
    #         else:
    #             uncond_tokens = [negative_prompt, negative_prompt_2]

    #         negative_prompt_embeds_list = []
    #         for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
    #             if isinstance(self, TextualInversionLoaderMixin):
    #                 negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

    #             max_length = prompt_embeds.shape[1]
    #             uncond_input = tokenizer(
    #                 negative_prompt,
    #                 padding="max_length",
    #                 max_length=max_length,
    #                 truncation=True,
    #                 return_tensors="pt",
    #             )

    #             negative_prompt_embeds = text_encoder(
    #                 uncond_input.input_ids.to(device),
    #                 output_hidden_states=True,
    #             )
    #             # We are only ALWAYS interested in the pooled output of the final text encoder
    #             negative_pooled_prompt_embeds = negative_prompt_embeds[0]
    #             negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

    #             negative_prompt_embeds_list.append(negative_prompt_embeds)

    #         negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    #     prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
    #     bs_embed, seq_len, _ = prompt_embeds.shape
    #     # duplicate text embeddings for each generation per prompt, using mps friendly method
    #     prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #     prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    #     if do_classifier_free_guidance:
    #         # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    #         seq_len = negative_prompt_embeds.shape[1]
    #         negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
    #         negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #         negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
    #     pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
    #         bs_embed * num_videos_per_prompt, -1
    #     )
    #     if do_classifier_free_guidance:
    #         negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
    #             bs_embed * num_videos_per_prompt, -1
    #         )

    #     class_tokens_mask = class_tokens_mask.to(device)
    #     return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, class_tokens_mask


    
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
    
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
            bs_embed * num_videos_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_videos_per_prompt).view(
                bs_embed * num_videos_per_prompt, -1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, single_model_length, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, single_model_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        # height // self.vae_scale_factor = 512 / 8 = 64
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # shape = [1, 4, 4, 64, 64]
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        single_model_length: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # Add parameters in PhotoMaker 
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0, # TODO: change to `style_strength_ratio` in the future
        class_tokens_mask: Optional[torch.LongTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        nc_flag = False,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        num_images_per_prompt: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        
        self._interrupt = False

        #
        if prompt_embeds is not None and class_tokens_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed. Make sure to generate `class_tokens_mask` from the same tokenizer that was used to generate `prompt_embeds`."
            )
        # check the input id images
        if input_id_images is None:
            raise ValueError(
                "Provide `input_id_images`. Cannot leave `input_id_images` undefined for PhotoMaker pipeline."
            )
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        num_id_images = len(input_id_images)
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        
        ## here is problem whether should we use the encoder_prompt_with_trigger_word or use self.encode_prompt
        
        negative_prompt_embeds_arr = []
        prompt_embeds_text_only_arr = []
        prompt_embeds_arr = []
        latents_arr = []
        add_time_ids_arr = []
        negative_pooled_prompt_embeds_arr = []
        pooled_prompt_embeds_text_only_arr = []
        pooled_prompt_embeds_arr = []
        
        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt = prompt,
            prompt_2 = prompt_2,
            device = device,
            num_id_images = num_id_images,
            prompt_embeds = prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            class_tokens_mask = class_tokens_mask,
            nc_flag = nc_flag,
        )
        # __import__("ipdb"  ).set_trace()
        
        # Additional: Encode input prompt without the trigger word for delayed conditioning
        # encode, remove trigger word token, then decode
        tokens_text_only = self.tokenizer.encode(prompt, add_special_tokens=False)
        trigger_word_token = self.tokenizer.convert_tokens_to_ids(self.trigger_word)
        if not nc_flag:
            tokens_text_only.remove(trigger_word_token)
        prompt_text_only = self.tokenizer.decode(tokens_text_only, add_special_tokens=False)
        
        print(prompt_text_only)
        
        (
            prompt_embeds_text_only, # [1, 77, 2048]
            negative_prompt_embeds, # [1, 77, 2048]
            pooled_prompt_embeds_text_only, # [1, 1280]# TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds, # [1, 1280]
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        
        # Additional: Prepare the input ID images
        dtype = next(self.id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values #[1, 3, 224, 224] --》 会被resize 到 224 悲伤
            
        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype)
        if not nc_flag:
            # Additional: Get the update text_embedding with the stacked ID embedding 
            print("id_pixel_values.dtype", id_pixel_values.dtype)
            print("prompt_embeds.dtype", prompt_embeds.dtype)
            # print("class_tokens_mask", class_tokens_mask)
            
            id_pixel_values = id_pixel_values.half() if id_pixel_values.dtype == torch.float32 else id_pixel_values
            self.id_encoder = self.id_encoder.half() 
            
            prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            pooled_prompt_embeds_arr.append(pooled_prompt_embeds)
            pooled_prompt_embeds = None
        negative_prompt_embeds_arr.append(negative_prompt_embeds)
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds_arr.append(negative_pooled_prompt_embeds)
        negative_pooled_prompt_embeds = None
        prompt_embeds_text_only_arr.append(prompt_embeds_text_only)
        prompt_embeds_text_only = None
        prompt_embeds_arr.append(prompt_embeds)
        prompt_embeds = None
        pooled_prompt_embeds_text_only_arr.append(pooled_prompt_embeds_text_only)
        pooled_prompt_embeds_text_only = None
        
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_arr, dim=0)
        print(negative_prompt_embeds.shape)
        if not nc_flag:
            prompt_embeds = torch.cat(prompt_embeds_arr, dim=0)
            print("prompt_embeds", prompt_embeds.shape)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_arr, dim=0)
            print("pooled_prompt_embeds", pooled_prompt_embeds.shape)
        prompt_embeds_text_only = torch.cat(prompt_embeds_text_only_arr, dim=0)
        print("prompt_embeds_text_only", prompt_embeds_text_only.shape)
        pooled_prompt_embeds_text_only = torch.cat(pooled_prompt_embeds_text_only_arr, dim=0)
        print("pool_prompt_embeds_text_only", pooled_prompt_embeds_text_only.shape)
        negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds_arr, dim=0)
        print("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds.shape)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # __import__('ipdb').set_trace()
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            single_model_length, 
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype if not nc_flag else prompt_embeds_text_only.dtype, 
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings 
        ## ！！ 同 phootmaker #10
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype if not nc_flag else prompt_embeds_text_only.dtype
        )

        if do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(1)
                # pooled_prompt_embeds_text_only = pooled_prompt_embeds_text_only.unsqueeze(1)
                if i <= start_merge_step or nc_flag: # 目前没有用处 start_merge_step = 0
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    )
                    print("negative_pooled_prompt_embeds", negative_pooled_prompt_embeds.shape)
                    print("pooled_prompt_embeds_text_only", pooled_prompt_embeds_text_only.shape)
                    print("pooled_prompt_embeds", pooled_prompt_embeds.shape)
                    
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0)
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    )
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                ts = torch.tensor([t], dtype=latent_model_input.dtype, device=latent_model_input.device)
                if do_classifier_free_guidance:
                    ts = ts.repeat(2)
                
                # print("latent_model_input", latent_model_input.shape)
                # print("prompt_embeds", prompt_embeds.shape)
                # print("ts", ts.shape)

                noise_pred = self.unet(
                    latent_model_input, # [2, 4, 4, 64, 64]
                    ts,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

            
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float32 and latents.dtype == torch.float16:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        #image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        image = ((image + 1) / 2).clamp(0, 1)
        video = rearrange(image, "(b f) c h w -> b c f h w", f=single_model_length).cpu()
        if not return_dict:
            return (video,)

        return AnimatePipelineOutput(videos=video)

  
    # Overrride to properly handle the loading and unloading of the additional text encoder.
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.
        
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)
        
        

    
    @validate_hf_hub_args
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        **kwargs,
    ):
        """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            weight_name (`str`):
                The weight name NOT the path to the weight.

            subfolder (`str`, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

            trigger_word (`str`, *optional*, defaults to `"img"`):
                The trigger word is used to identify the position of class word in the text prompt,
                and it is recommended not to set it as a common word.
                This trigger word must be placed after the class word when used, otherwise, it will affect the performance of the personalized generation.
        """

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        self.trigger_word = trigger_word
        # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
        print(f"Loading PhotoMaker components [1] id_encoder from [{pretrained_model_name_or_path_or_dict}]...")
        id_encoder = PhotoMakerIDEncoder()
        id_encoder.load_state_dict(state_dict["id_encoder"], strict=False)
        id_encoder = id_encoder.to(self.device, dtype=self.unet.dtype)
        self.id_encoder = id_encoder
        self.id_image_processor = CLIPImageProcessor()

        # load lora into models
        print(f"Loading PhotoMaker components [2] lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word token
        if self.tokenizer is not None:
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)

        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)


    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        nc_flag: bool = False,
    ):
        device = device or self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders): # prompts = ['a', 'a' ] ???
                input_ids = tokenizer.encode(prompt) # TODO: batch encode
                # input_ids = [49406, 320, 49407]
                clean_index = 0
                clean_input_ids = []
                class_token_index = []
                # Find out the corresponding class word token based on the newly added trigger word token
                
                # __import__("ipdb").set_trace()
                for i, token_id in enumerate(input_ids):
                    if token_id == image_token_id:
                        class_token_index.append(clean_index - 1)
                    else:
                        clean_input_ids.append(token_id)
                        clean_index += 1
                if nc_flag:
                    return None, None, None
                if len(class_token_index) > 1:
                    raise ValueError(
                        f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                            Trigger word: {self.trigger_word}, Prompt: {prompt}."
                    )
                # elif len(class_token_index) == 0 and not nc_flag:
                #     raise ValueError(
                #         f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                #             Trigger word: {self.trigger_word}, Prompt: {prompt}."
                #     )
                print("class_token_index", class_token_index)
                class_token_index = class_token_index[0]

                # Expand the class word token and corresponding mask
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + \
                    clean_input_ids[class_token_index+1:]

                # Truncation or padding
                max_len = tokenizer.model_max_length
                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                        max_len - len(clean_input_ids)
                    )

                class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False \
                     for i in range(len(clean_input_ids))]

                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                prompt_embeds = text_encoder(
                    clean_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask

    @property
    def interrupt(self):
        return self._interrupt
