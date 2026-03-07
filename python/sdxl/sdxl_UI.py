import subprocess
import sys

print("Install missing libraries if necessary")
subprocess.run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "gradio",
    "pillow"
], check=True)

import os
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


# ----- device selection -----
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}")


# ----- load models once -----
print("Load text-to-image model")
pipe_txt2img = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    use_safetensors=True,
)

print("Load image-to-image model")
pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    use_safetensors=True,
)

if device in ["cuda", "cpu"]:
    pipe_txt2img.enable_attention_slicing()
    pipe_txt2img.enable_vae_slicing()
    pipe_img2img.enable_attention_slicing()
    pipe_img2img.enable_vae_slicing()

pipe_txt2img = pipe_txt2img.to(device)
pipe_img2img = pipe_img2img.to(device)

print("Create output directory")
out_dir = "sdxl_outputs"
os.makedirs(out_dir, exist_ok=True)


def generate_image(prompt, negative_prompt, steps, guidance, width, height, input_image, strength):
    width = int(width)
    height = int(height)
    steps = int(steps)
    guidance = float(guidance)
    strength = float(strength)

    # ----- image-to-image -----
    if input_image is not None:
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        input_image = input_image.convert("RGB").resize((width, height))

        result = pipe_img2img(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
        )
        image = result.images[0]
        mode = "img2img"

    # ----- text-to-image -----
    else:
        result = pipe_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        image = result.images[0]
        mode = "txt2img"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(out_dir, f"sdxl_{mode}_{timestamp}.png")
    txt_path = os.path.join(out_dir, f"sdxl_{mode}_{timestamp}.txt")

    image.save(img_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MODEL: stabilityai/stable-diffusion-xl-base-1.0\n")
        f.write(f"MODE: {mode}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"DEVICE: {device}\n")
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"NEGATIVE_PROMPT: {negative_prompt}\n")
        f.write(f"STEPS: {steps}\n")
        f.write(f"GUIDANCE_SCALE: {guidance}\n")
        f.write(f"SIZE: {width}x{height}\n")
        if input_image is not None:
            f.write(f"STRENGTH: {strength}\n")

    return image, f"Saved image: {img_path}\nSaved prompt info: {txt_path}"


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            lines=3,
            placeholder="Describe the image you want to generate"
        ),
        gr.Textbox(
            label="Negative prompt",
            lines=2,
            value="blurry, low quality, distorted, extra fingers, duplicate objects, text, watermark"
        ),
        gr.Slider(
            minimum=10,
            maximum=50,
            step=1,
            value=15,
            label="Number of inference steps",
            info="Number of denoising steps. More steps usually improve quality, but generation takes longer."
        ),
        gr.Slider(
            minimum=1.0,
            maximum=12.0,
            step=0.5,
            value=7.5,
            label="Guidance scale",
            info="How strongly the model follows the prompt. Low values are looser and more creative; high values are more literal."
        ),
        gr.Slider(
            minimum=256,
            maximum=1024,
            step=64,
            value=768,
            label="Image width",
            info="Width in pixels. Must be divisible by 64."
        ),
        gr.Slider(
            minimum=256,
            maximum=1024,
            step=64,
            value=768,
            label="Image height",
            info="Height in pixels. Must be divisible by 64."
        ),
        gr.Image(type="pil", label="Input image (optional)"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.05,
            value=0.6,
            label="Transformation strength",
            info="Only used when an input image is provided. Low = closer to the original image, high = more transformed."
        )
    ],
    outputs=[
        gr.Image(label="Generated image"),
        gr.Textbox(label="Saved files")
    ],
    title="Evolution of AI: Stable Diffusion XL Generator",
    description="""
Generate images using Stable Diffusion XL.

- No input image: text-to-image
- With input image: image-to-image

**Inference steps**
The model gradually removes noise to form the final image.
More steps usually improve quality but make generation slower.

**Guidance scale**
Controls how strongly the model follows the prompt.
Lower values allow more freedom; higher values follow the text more strictly.

**Transformation strength**
Used only for image-to-image.
Low values preserve the original image more.
High values transform it more strongly.
"""
)

demo.launch(inbrowser=True)