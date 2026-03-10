import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline

# -----------------------------
# Configuration
# -----------------------------
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "./my_sdxl_lora"   # folder produced by training
LORA_WEIGHT_NAME = None        # e.g. "pytorch_lora_weights.safetensors" if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

pipe = None


# -----------------------------
# Model loading
# -----------------------------
def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    print(f"Loading base model on {DEVICE}...")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )

    if DEVICE == "cuda":
        pipe = pipe.to("cuda")

    print("Loading LoRA weights...")
    if LORA_WEIGHT_NAME:
        pipe.load_lora_weights(LORA_PATH, weight_name=LORA_WEIGHT_NAME, adapter_name="epfl_style")
    else:
        pipe.load_lora_weights(LORA_PATH, adapter_name="epfl_style")

    # Activate adapter with default strength
    pipe.set_adapters(["epfl_style"], adapter_weights=[1.0])

    return pipe


# -----------------------------
# Generation
# -----------------------------
def generate_images(
    prompt,
    negative_prompt,
    steps,
    guidance,
    lora_strength,
    width,
    height,
    num_images,
    seed,
):
    p = load_pipeline()

    # Update LoRA strength dynamically
    p.set_adapters(["epfl_style"], adapter_weights=[float(lora_strength)])

    # Seed handling
    if seed == -1:
        generator = None
        used_seed = "random"
    else:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        used_seed = str(seed)

    result = p(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=int(width),
        height=int(height),
        num_images_per_prompt=int(num_images),
        generator=generator,
    )

    images = result.images

    info = (
        f"Prompt: {prompt}\n"
        f"Negative prompt: {negative_prompt}\n"
        f"Steps: {steps}\n"
        f"Guidance: {guidance}\n"
        f"LoRA strength: {lora_strength}\n"
        f"Size: {width}x{height}\n"
        f"Images: {num_images}\n"
        f"Seed: {used_seed}"
    )

    return images, info


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="SDXL LoRA – EPFL+ECAL Lab Style") as demo:
    gr.Markdown("# SDXL LoRA – EPFL+ECAL Lab Style")
    gr.Markdown(
        "Generate portraits with your fine-tuned LoRA.\n\n"
        "Start with the style token in the prompt, for example:\n"
        "`epfl_ecal_lab_style portrait of a person, studio photography`"
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="epfl_ecal_lab_style portrait of a person, studio photography",
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                value="blurry, low quality, distorted face, deformed, bad anatomy",
            )

            steps = gr.Slider(
                minimum=10,
                maximum=60,
                value=30,
                step=1,
                label="Inference Steps",
            )

            guidance = gr.Slider(
                minimum=1.0,
                maximum=12.0,
                value=7.0,
                step=0.5,
                label="Guidance Scale",
            )

            lora_strength = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=1.0,
                step=0.05,
                label="LoRA Strength",
            )

            with gr.Row():
                width = gr.Slider(
                    minimum=512,
                    maximum=1536,
                    value=1024,
                    step=64,
                    label="Width",
                )
                height = gr.Slider(
                    minimum=512,
                    maximum=1536,
                    value=1024,
                    step=64,
                    label="Height",
                )

            with gr.Row():
                num_images = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Number of Images",
                )
                seed = gr.Slider(
                    minimum=-1,
                    maximum=999999,
                    value=-1,
                    step=1,
                    label="Seed (-1 = random)",
                )

            run_button = gr.Button("Generate")

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
            info = gr.Textbox(label="Generation Info", lines=10)

    run_button.click(
        fn=generate_images,
        inputs=[
            prompt,
            negative_prompt,
            steps,
            guidance,
            lora_strength,
            width,
            height,
            num_images,
            seed,
        ],
        outputs=[gallery, info],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)