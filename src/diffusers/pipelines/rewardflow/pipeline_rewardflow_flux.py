# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable

import numpy as np
import PIL
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from ...loaders import FluxRewardFlowLoraLoaderMixin
from ...models import AutoencoderKLFluxRewardFlow, FluxRewardFlowTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .image_processor import FluxRewardFlowImageProcessor
from .pipeline_output import FluxRewardFlowPipelineOutput
from .rewards import (
    GroundingDINOReward,
    Qwen25VLCaptionReward,
    RegionCLIPReward,
    RewardGuidance,
    SigLIPReward,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxRewardFlowPipeline

        >>> pipe = FluxRewardFlowPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.2--base-9B", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=50, guidance_scale=4.0).images[0]
        >>> image.save("FluxRewardFlow_output.png")
        ```
"""


# Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.compute_empirical_mu
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FluxRewardFlowPipeline(DiffusionPipeline, FluxRewardFlowLoraLoaderMixin):
    r"""
    The FluxRewardFlow  pipeline for text-to-image generation.

    Reference:
    [https://bfl.ai/blog/FluxRewardFlow--towards-interactive-visual-intelligence](https://bfl.ai/blog/FluxRewardFlow--towards-interactive-visual-intelligence)

    Args:
        transformer ([`FluxRewardFlowTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFluxRewardFlow`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3ForCausalLM`]):
            [Qwen3ForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)
        tokenizer (`Qwen2TokenizerFast`):
            Tokenizer of class
            [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFluxRewardFlow,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: FluxRewardFlowTransformer2DModel,
        is_distilled: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.register_to_config(is_distilled=is_distilled)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = FluxRewardFlowImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128
        self._reward_fns = self._get_default_reward_fns(
            device=self.text_encoder.device, dtype=self.text_encoder.dtype
        )

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: str | list[str],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: list[int] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._prepare_text_ids
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: torch.Tensor | None = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._prepare_latent_ids
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._prepare_image_ids
    def _prepare_image_ids(
        image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):


        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._pack_latents
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._unpack_latents_with_ids
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _decode_latents_for_reward(self, latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pt")
        return image

    def _apply_reward_guidance(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt: str | list[str],
        reward_guidance: RewardGuidance,
        reward_guidance_scale: float,
        reward_guidance_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        last_reward_values = None
        last_reward_weights = None

        latents_guided = latents
        if reward_guidance_steps <= 0:
            return latents_guided, last_reward_values, last_reward_weights
        import time
        for _ in range(reward_guidance_steps):
            latents_guided = latents_guided.detach().requires_grad_(True)
            with torch.enable_grad():
                time.sleep(3)  # to simulate the time taken by reward computation and make the effect of reward guidance more pronounced in testing 
                image = self._decode_latents_for_reward(latents_guided, latent_ids)
                total_reward, reward_values, reward_weights = reward_guidance.compute(image=image, prompt=prompt)
                last_reward_values = reward_values
                last_reward_weights = reward_weights

                if not total_reward.requires_grad:
                    logger.warning_once(
                        "Skipping reward guidance update because `total_reward` is not differentiable. "
                        "Ensure at least one reward function preserves gradients from image to reward."
                    )
                    return latents_guided.detach(), last_reward_values, last_reward_weights

                grad = torch.autograd.grad(
                    total_reward, latents_guided, retain_graph=False, allow_unused=True
                )[0]

            if grad is None:
                logger.warning_once(
                    "Skipping reward guidance update because reward has no gradient with respect to latents."
                )
                return latents_guided.detach(), last_reward_values, last_reward_weights

            grad_norm = torch.norm(grad, p=2)
            grad = grad / (grad_norm + 1e-6)
            latents_guided = (latents_guided + reward_guidance_scale * grad).detach()

        # print("reward guided", total_reward, last_reward_weights)

        return latents_guided, last_reward_values, last_reward_weights

    def _get_default_reward_fns(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
        model_ids: dict[str, str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        use_cache = model_ids is None and model_kwargs is None
        cached_device = getattr(self, "_reward_fns_device", None)
        cached_fns = getattr(self, "_reward_fns", None)
        if use_cache and cached_fns is not None and cached_device == device:
            return cached_fns

        model_ids = model_ids or {}
        model_kwargs = model_kwargs or {}

        def _select_kwargs(key: str) -> dict[str, Any]:
            if not model_kwargs:
                return {}
            if isinstance(model_kwargs.get(key), dict):
                merged = {}
                if isinstance(model_kwargs.get("default"), dict):
                    merged.update(model_kwargs["default"])
                merged.update(model_kwargs[key])
                return merged
            if isinstance(model_kwargs.get("default"), dict):
                return dict(model_kwargs["default"])
            if all(not isinstance(v, dict) for v in model_kwargs.values()):
                return dict(model_kwargs)
            return {}

        def _get_id(key: str, default: str):
            return model_ids.get(key, model_ids.get("qwen3_vl" if key == "qwen_vl" else key, default))

        reward_fns = []
        try:
            reward_fns.append(
                SigLIPReward(
                    model_ids.get("siglip", "google/siglip-so400m-patch14-384"),
                    device=device,
                    dtype=dtype,
                    **_select_kwargs("siglip"),
                )
            )
        except Exception as exc:
            print("Skipping RegionCLIPReward: {exc}")

        try:
            reward_fns.append(
                RegionCLIPReward(
                    model_ids.get("region_clip", "openai/clip-vit-large-patch14"),
                    device=device,
                    dtype=dtype,
                    **_select_kwargs("region_clip"),
                )
            )
        except Exception as exc:
            print("Skipping RegionCLIPReward: {exc}")
        if not reward_fns:
            raise ValueError("No reward functions could be initialized.")
        if use_cache:
            self._reward_fns = reward_fns
            self._reward_fns_device = device
        return reward_fns

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    # Copied from diffusers.pipelines.FluxRewardFlow.pipeline_FluxRewardFlow.FluxRewardFlowPipeline.prepare_image_latents
    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(imagge_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self.config.is_distilled

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: list[PIL.Image.Image] | PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: str | list[str] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        start_reward: int = 3,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
        reward_guidance: bool = False,
        reward_guidance_scale: float = 1.0,
        reward_guidance_steps: int = 2,
        reward_guidance_temperature: float = 1.0,
        reward_fns: list[Callable] | None = None,
        reward_model_ids: dict[str, str] | None = None,
        reward_model_kwargs: dict[str, Any] | None = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. For step-wise distilled models,
                `guidance_scale` is ignored.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Note that "" is used as the negative prompt in this pipeline.
                If not provided, will be generated from "".
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            text_encoder_out_layers (`tuple[int]`):
                Layer indices to use in the `text_encoder` to derive the final prompt embeddings.
            reward_guidance (`bool`, *optional*, defaults to `False`):
                Whether to apply reward-guided updates to latents at each diffusion step.
            reward_guidance_scale (`float`, *optional*, defaults to `1.0`):
                Step size for reward-guided latent updates.
            reward_guidance_steps (`int`, *optional*, defaults to `4`):
                Number of reward optimization steps to run per diffusion step.
            reward_guidance_temperature (`float`, *optional*, defaults to `1.0`):
                Softmax temperature used to compute dynamic reward weights.
            reward_fns (`List[Callable]`, *optional*):
                List of reward functions. Each callable must accept `(image, prompt)` and return a scalar tensor.
                If `None`, a default set is used: SigLIP, RegionCLIP, and Qwen3-VL-1B caption reward.
            reward_model_ids (`dict[str, str]`, *optional*):
                Override model IDs or local paths for reward models. Keys: `siglip`, `region_clip`, `qwen3_vl`.
            reward_model_kwargs (`dict[str, Any]`, *optional*):
                Extra kwargs forwarded to reward model `from_pretrained` calls (e.g. `cache_dir`, `local_files_only`,
                `token`, `revision`). Can be a flat dict for all models or a dict with `default` and per-model keys.

        Examples:

        Returns:
            [`~pipelines.FluxRewardFlow.FluxRewardFlowPipelineOutput`] or `tuple`: [`~pipelines.FluxRewardFlow.FluxRewardFlowPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._guidance_scale = guidance_scale if reward_guidance else 1.0
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        reward_guidance_module = None
        reward_prompt = prompt
        if reward_guidance:
            if reward_prompt is None:
                raise ValueError("`reward_guidance=True` requires `prompt` to be provided.")
            if reward_fns is None:
                reward_fns = self._get_default_reward_fns(
                    device=device,
                    dtype=self.text_encoder.dtype,
                    model_ids=reward_model_ids,
                    model_kwargs=reward_model_kwargs,
                )
            if reward_fns is None:
                raise ValueError("`reward_guidance=True` requires `reward_fns` to be provided.")
            reward_guidance_module = RewardGuidance(reward_fns, temperature=reward_guidance_temperature)
            for reward_fn in reward_fns:
                prepare_fn = getattr(reward_fn, "prepare", None)
                if callable(prepare_fn):
                    prepare_fn(prompt=reward_prompt, device=device)

        if not reward_guidance and image != None:
            raise ValueError("Passing `image` is not supported when `reward_guidance=False`. Please set `reward_guidance=True` to use image conditioning.")
        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width
        height = height or self.default_sample_size * self.vae_scale_factor
        # if image != None:
        #     reward_guidance=False
        num_channels_latents = self.transformer.config.in_channels // 4
        width = width or self.default_sample_size * self.vae_scale_factor
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        _, con, _ = latents.shape
        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,  # (B, image_seq_len, C)
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,  # B, text_seq_len, 4
                        img_ids=latent_image_ids,  # B, image_seq_len, 4
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                    
                if reward_guidance and i >= start_reward:
                    latents, _, _ = self._apply_reward_guidance(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt=reward_prompt,
                        reward_guidance=reward_guidance_module,
                        reward_guidance_scale=reward_guidance_scale,
                        reward_guidance_steps=reward_guidance_steps,
                    )
                    

                
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxRewardFlowPipelineOutput(images=image)
