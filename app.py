import os
import random
from pathlib import Path

import gradio as gr
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
import torch
from dotenv import load_dotenv
from huggingface_hub import login


def generate(
    prompt,
    negative_prompt,
    width,
    height,
    seed,
    batch_count,
    scheduler,
    steps,
    cfg_scale,
    progress=gr.Progress(),
):
    seed = int(seed) if seed != 0 else random.randint(0, 1_000_000_000_000)
    generator = torch.manual_seed(seed)

    if scheduler == "EulerDiscreteScheduler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if scheduler == "DPMSolverMultistepScheduler":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    prompt_dir = (prompt[:200] + "..") if len(prompt) > 200 else prompt
    prompt_dir = prompt_dir.replace(" ", "_")
    Path(f"outputs_gradio/{prompt_dir}").mkdir(parents=True, exist_ok=True)

    list_paths = []
    use_refiner = False
    for i in progress.tqdm(range(batch_count)):
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            output_type="latent" if use_refiner else "pil",
        ).images[0]
        # if use_refiner:
            # image = refiner(prompt=prompt, image=image[None, :]).images[0]
        image.save(f"outputs_gradio/{prompt_dir}/image_{i+1}_{seed}.png")
        list_paths.append(f"outputs_gradio/{prompt_dir}/image_{i+1}_{seed}.png")

    return list_paths, list_paths


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Demo")
    with gr.Tab("text-to-image"):
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Textbox(
                    value="stabilityai/stable-diffusion-xl-base-1.0",
                    label="Model",
                    interactive=False,
                    show_label=True,
                )
                prompt = gr.Textbox(
                    value="",
                    label="Prompt",
                    show_label=True,
                )
                negative_prompt = gr.Textbox(
                    value="",
                    label="Negative Prompt",
                    show_label=True,
                )
                width = gr.Slider(
                    minimum=128, maximum=1024, value=1024, step=64, label="Width"
                )
                height = gr.Slider(
                    minimum=128, maximum=1024, value=1024, step=64, label="Height"
                )
                with gr.Row():
                    seed = gr.Slider(
                        value=0,
                        minimum=0,
                        maximum=1_000_000_000_000,
                        step=1,
                        label="Seed",
                    )
                    batch_count = gr.Slider(
                        value=4, minimum=1, maximum=10, step=1, label="Batch Count"
                    )
                with gr.Row():
                    scheduler = gr.Dropdown(
                        ["EulerDiscreteScheduler", "DPMSolverMultistepScheduler"],
                        value="DPMSolverMultistepScheduler",
                        label="Scheduler",
                    )
                    steps = gr.Slider(
                        value=40, minimum=1, maximum=130, step=1, label="Steps"
                    )
                cfg_scale = gr.Slider(
                    value=7.5, minimum=1, maximum=30, step=0.5, label="CFG Scale"
                )

                btn1 = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    columns=4, rows=4, height="auto", object_fit="contain"
                )
                files = gr.Files()

    btn1.click(
        generate,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            seed,
            batch_count,
            scheduler,
            steps,
            cfg_scale,
        ],
        outputs=[gallery, files],
    )

if __name__ == "__main__":

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        # custom_pipeline="lpw_stable_diffusion",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda")
    # pipe.load_lora_weights("lora-trained-xl-1.4/checkpoint-200")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    

    # refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-refiner-1.0",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",
    # )
    # refiner.to("cuda")
    demo.queue(concurrency_count=1)
    demo.launch()
