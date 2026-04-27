import torch
from diffusers import FluxRewardFlowPipeline
from PIL import Image

seed = 42
torch.manual_seed(seed)          # CPU + current CUDA device RNG

device = "cuda"
dtype = torch.bfloat16

model_dir = "<Path to your model directory>"

qwen_device = None
reward_model_kwargs = {
    "qwen_vl": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": dtype,
        "offload_to_cpu": True,
        "onload_device": device,
    }
}

if qwen_device is not None:
    reward_model_kwargs["qwen_vl"]["device_map"] = {"": device}

pipe = FluxRewardFlowPipeline.from_pretrained(
    model_dir,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
    local_files_only=True,
)
pipe.to(device)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A happy puppy with soft fur and bright blue eyes jumping after a toy, in a cozy living room"
# prompt = "A television made of water that displays an image of a cityscape at night"




image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=12,
    num_inference_steps=45,
    reward_guidance=True,
    reward_guidance_scale=5.0,
    reward_guidance_steps=5,
    start_reward=3,
    reward_model_kwargs=reward_model_kwargs,
).images[0]
image.resize((512, 512)).save("image_edited.png")