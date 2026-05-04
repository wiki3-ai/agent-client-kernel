from diffusers import StableDiffusionPipeline
import torch

print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")

print("Generating cat image...")
prompt = "An adorable fluffy cat with bright curious eyes, soft studio lighting, clean background, high-quality pet photography style"
image = pipe(prompt).images[0]

output_path = "/workspaces/agent-client-kernel/cat.png"
image.save(output_path)
print(f"Image saved to: {output_path}")
