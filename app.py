from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline, UnCLIPImageVariationPipeline
from diffusers import LDMSuperResolutionPipeline, StableDiffusionUpscalePipeline
from transformers import CLIPVisionModelWithProjection, CLIPFeatureExtractor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

image_upscaler_model_id = "CompVis/ldm-super-resolution-4x-openimages"
image_upscaler = LDMSuperResolutionPipeline.from_pretrained(image_upscaler_model_id, torch_dtype=torch.float16)
image_upscaler = image_upscaler.to(device)


image_generator = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
image_generator = image_generator.to(device)

image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",torch_dtype=torch.float16)
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
variation_generator = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", image_encoder=image_encoder, feature_extractor=feature_extractor, torch_dtype=torch.float16)
variation_generator = variation_generator.to(device)

import gradio as gr

# history = []
# history_index = 0

def generate_images(prompt, negative_prompt="", num_images_per_prompt=4, prior_num_inference_steps=50, decoder_num_inference_steps=25, super_res_num_inference_steps=7, prior_guidance_scale=4.0, decoder_guidance_scale=4.0, manual_seed=-1, history=[], history_index=0):
    generator = None
    if manual_seed > -1:
        generator = torch.Generator()
        generator.manual_seed(int(manual_seed))
    
    # generate
    new_images = image_generator([prompt], negative_prompt=negative_prompt, num_images_per_prompt=int(num_images_per_prompt), prior_num_inference_steps=int(prior_num_inference_steps), decoder_num_inference_steps=int(decoder_num_inference_steps), super_res_num_inference_steps=int(super_res_num_inference_steps), prior_guidance_scale=float(prior_guidance_scale), decoder_guidance_scale=float(decoder_guidance_scale), generator=generator)
    new_images = new_images['images']
    
    # update history
    new_history, new_history_index = get_appended_history(history, history_index, new_images)
    
    # empty cache
    torch.cuda.empty_cache()

    return new_images, new_history, new_history_index

def generate_variations(images, prompt="", negative_prompt="", selected_image_index=1, num_images_per_prompt=4, decoder_num_inference_steps=25, decoder_guidance_scale=4.0, history=[], history_index=0):
    if len(images) < 1:
        return []
    
    # get target image
    target_image = Image.open(images[selected_image_index - 1]['name'])

    # generate
    new_images = variation_generator(image=target_image, prompt=prompt, negative_prompt=negative_prompt, num_images_per_prompt=int(num_images_per_prompt), decoder_num_inference_steps=int(decoder_num_inference_steps), decoder_guidance_scale=float(decoder_guidance_scale), image_embeddings=None).images

    # update history
    new_history, new_history_index = get_appended_history(history, history_index, new_images)

    # empty cache
    torch.cuda.empty_cache()

    return new_images, new_history, new_history_index

def get_appended_history(history, history_index, new_images):
    history.append(new_images)
    history_index = len(history) - 1
    return history, history_index

def step_back_in_history(history, history_index):
    if len(history) < 1:
        return [], history, history_index
    if history_index - 1 > -1:
        history_index -= 1
    return history[history_index], history, history_index

def step_forward_in_history(history, history_index):
    if len(history) < 1:
        return [], history, history_index
    if history_index + 1 < len(history):
        history_index += 1
    return history[history_index], history, history_index

def highlight_selected_image(demo_block, selected_image_index):
    print(demo_block)
    # new_css = "gallery button:nth-of-type(TARGET_IMAGE_INDEX) {border: 6px solid blue}"
    # new_css = new_css.replace("TARGET_IMAGE_INDEX", str(selected_image_index))
    # demo_block.css = new_css

def upscale_selected_image(images, selected_image_index=1, upscale_tab=None):
    if len(images) < 1:
        return None
    # get target image
    target_image = Image.open(images[selected_image_index - 1]['name'])
    upscaled_image = image_upscaler(image=target_image.resize((196,196)), num_inference_steps=100).images
    torch.cuda.empty_cache()
    return upscaled_image[0], gr.Tabs.update(selected=1)

from time import time
import os

def save_upscaled_image(upscaled_image, save_dir):
    if upscaled_image is None:
        return None
    timestamp = int(time())
    file_name = f"{timestamp}.png"
    save_path = os.path.join(save_dir, file_name)
    upscaled_image.save(save_path)

with gr.Blocks() as demo:
    history = gr.State([])
    history_index = gr.State(0)
    # print(demo)
    # demo.css = "#gallery button:first-of-type {border: 2px solid blue}"
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(value="Beautiful painting of a ghastly skeleton king in armor emerging from a medieval crypt by Frank Frazetta. Foreboding mood. uhd, high resolution, fine details, intricate details, centered, high quality, fine details, masterpiece, fine brushstrokes, dynamic movement", interactive=True, label="Prompt")
            negative_prompt = gr.Textbox(value="low quality, mid quality, low resolution, jpeg artifacts, watermark, text, poorly drawn, amateur", interactive=True, label="Negative Prompt")
            decoder_guidance_scale_ig = gr.Slider(1,10,step=0.5, value=5.0, interactive=True, label="Guidance Scale For Image Generation")
            with gr.Accordion('More Options', open=False):
                num_images_per_prompt = gr.Slider(1,8,step=1,value=2, label="Number of Images", visible=True, interactive=False)
                prior_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Prior Inference Steps", visible=True, interactive=False)
                decoder_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Decoder Inference Steps", visible=True, interactive=False)
                super_res_num_inference_steps = gr.Slider(1,20,step=1,value=7, label="Superres Inference Steps", visible=True, interactive=False)
                prior_guidance_scale = gr.Slider(1,10,step=0.5, value=4.0, label="Prior Guidance Scale", visible=True, interactive=False)
                manual_seed = gr.Number(value=-1, interactive=True, label="Seed")
        with gr.Column(scale=1):
            with gr.Tabs() as tabs:
                with gr.TabItem("Generated", id=0):
                    gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery").style(grid=[2], height="auto", elem_id="main-gallery", container=True)
                    with gr.Row():
                        back_in_history_button = gr.Button(value="Back in History", variant="secondary")
                        back_in_history_button.click(step_back_in_history, inputs=[history, history_index], outputs=[gallery, history, history_index], queue=False)

                        forward_in_history_button = gr.Button(value="Forward in History", variant="secondary")
                        forward_in_history_button.click(step_forward_in_history, inputs=[history, history_index], outputs=[gallery, history, history_index], queue=False)

                        # gallery.change(fn=lambda value: gr.update(value=value), inputs=history_index, outputs=test)
                    with gr.Row():
                        generate_button = gr.Button(value="Generate New Images", variant="primary")

                    with gr.Row():
                        selected_image_index = gr.Radio([1,2], value=1, interactive=True, label="Selected Image for Variations")
                        decoder_guidance_scale_vg = gr.Slider(1,10,step=0.5, value=5.5, interactive=True, label="Guidance Scale For Variation Generation")

                    # selected_image_index.change(highlight_selected_image, inputs=[demo, selected_image_index], outputs=[])
                    with gr.Row():
                        upscale_button = gr.Button(value="Upscale Selected Image", variant="secondary")
                        variations_button = gr.Button(value=f"Generate Variations from Selected Image", variant="primary")

                    variations_button.click(generate_variations, inputs=[gallery, prompt, negative_prompt, selected_image_index, num_images_per_prompt, decoder_num_inference_steps, decoder_guidance_scale_vg, history, history_index], outputs=[gallery, history, history_index])

                    generate_button.click(generate_images, inputs=[prompt, negative_prompt, num_images_per_prompt, prior_num_inference_steps, decoder_num_inference_steps, super_res_num_inference_steps, prior_guidance_scale, decoder_guidance_scale_ig, manual_seed, history, history_index], outputs=[gallery, history, history_index])
                with gr.TabItem('Upscaled', id=1) as upscale_tab:
                    with gr.Row():
                        upscaled_image = gr.Image(type="pil", shape=(768,None))

                    with gr.Row():
                        save_path = gr.Textbox(value="./outputs/upscaled", interactive=True, label="Save Path", visible=False)
                        save_upscaled_image_button = gr.Button(value="Save Upscaled Image", variant="primary")

                    save_upscaled_image_button.click(save_upscaled_image, inputs=[upscaled_image, save_path], outputs=[])
                    upscale_button.click(upscale_selected_image, inputs=[gallery, selected_image_index], outputs=[upscaled_image, tabs])

            
# demo.queue(concurrency_count=2)
# demo.launch(share=True)

demo.launch()