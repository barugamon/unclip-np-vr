from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline, UnCLIPImageVariationPipeline
from diffusers import LDMSuperResolutionPipeline, StableDiffusionUpscalePipeline
from transformers import CLIPVisionModelWithProjection, CLIPFeatureExtractor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

image_upscaler_model_id = "CompVis/ldm-super-resolution-4x-openimages"
image_upscaler = LDMSuperResolutionPipeline.from_pretrained(image_upscaler_model_id, torch_dtype=torch.float16)
image_upscaler = image_upscaler.to(device)


txt2img_generator = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
txt2img_generator = txt2img_generator.to(device)

image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",torch_dtype=torch.float16)
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
img2img_generator = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", image_encoder=image_encoder, feature_extractor=feature_extractor, torch_dtype=torch.float16)
img2img_generator = img2img_generator.to(device)

# txt2img_generator.enable_sequential_cpu_offload()
# img2img_generator.enable_sequential_cpu_offload()


import gradio as gr

@torch.no_grad()
def do_generate_txt2img(prompt, negative_prompt="", num_images_per_prompt=4, prior_num_inference_steps=50, decoder_num_inference_steps=25, super_res_num_inference_steps=7, prior_guidance_scale=4.0, decoder_guidance_scale=4.0, manual_seed=-1, history=[], history_index=0):
    generator = None
    if manual_seed > -1:
        generator = torch.Generator()
        generator.manual_seed(int(manual_seed))
    
    # generate
    new_images = txt2img_generator([prompt], negative_prompt=negative_prompt, num_images_per_prompt=int(num_images_per_prompt), prior_num_inference_steps=int(prior_num_inference_steps), decoder_num_inference_steps=int(decoder_num_inference_steps), super_res_num_inference_steps=int(super_res_num_inference_steps), prior_guidance_scale=float(prior_guidance_scale), decoder_guidance_scale=float(decoder_guidance_scale), generator=generator)
    new_images = new_images['images']
    
    # update history
    new_history, new_history_index = do_get_appended_history(history, history_index, new_images)
    
    # empty cache
    torch.cuda.empty_cache()

    return new_images, new_history, new_history_index

@torch.no_grad()
def do_generate_img2img(target_image, prompt="", negative_prompt="", num_images_per_prompt=4, decoder_num_inference_steps=25, decoder_guidance_scale=4.0, manual_seed=-1, history=[], history_index=0):
    generator = None
    if manual_seed > -1:
        generator = torch.Generator()
        generator.manual_seed(int(manual_seed))
    # resize
    target_image = target_image.resize((256,256))
    # generate
    new_images = img2img_generator(image=target_image, prompt=prompt, negative_prompt=negative_prompt, num_images_per_prompt=int(num_images_per_prompt), decoder_num_inference_steps=int(decoder_num_inference_steps), decoder_guidance_scale=float(decoder_guidance_scale), image_embeddings=None, generator=generator).images

    # update history
    new_history, new_history_index = do_get_appended_history(history, history_index, new_images)

    # empty cache
    torch.cuda.empty_cache()

    return new_images, new_history, new_history_index

def do_get_appended_history(history, history_index, new_images):
    history.append(new_images)
    history_index = len(history) - 1
    return history, history_index

def do_step_back_in_history(history, history_index):
    if len(history) < 1:
        return [], history, history_index
    if history_index - 1 > -1:
        history_index -= 1
    return history[history_index], history, history_index

def do_step_forward_in_history(history, history_index):
    if len(history) < 1:
        return [], history, history_index
    if history_index + 1 < len(history):
        history_index += 1
    return history[history_index], history, history_index

def highlight_selected_image(selected_image_index):
    new_css = "gallery button:nth-of-type(TARGET_IMAGE_INDEX) {border: 6px solid blue}"
    new_css = new_css.replace("TARGET_IMAGE_INDEX", str(selected_image_index))
    return gr.update(css=new_css)

@torch.no_grad()
def do_upscale_image(target_image, downsample_size=128, num_inference_steps=100):
    upscaled_image = image_upscaler(image=target_image.resize((downsample_size, downsample_size)), num_inference_steps=num_inference_steps).images
    torch.cuda.empty_cache()
    return upscaled_image[0]

from time import time
import os

def do_send_image_from_history(history, history_index, image_index, tab_index=0):
    if image_index < 1 or (image_index - 1) > len(history[history_index]):
        return None
    return history[history_index][image_index-1], gr.Tabs.update(selected=tab_index)

def do_send_image_from_history_with_prompts(history, history_index, image_index, tab_index=0, prompt="", negative_prompt=""):
    if image_index < 1 or (image_index - 1) > len(history[history_index]):
        return None
    return history[history_index][image_index-1], gr.Tabs.update(selected=tab_index), prompt, negative_prompt

def do_update_selected_tab_index(new_tab_index):
    return gr.update(selected=new_tab_index)

def do_save_image(target_image, save_dir='./outputs/upscaled'):
    if target_image is None:
        return None
    timestamp = int(time())
    file_name = f"{timestamp}.png"
    save_path = os.path.join(save_dir, file_name)
    target_image.save(save_path)

with gr.Blocks(css="./style.css") as demo:
    txt2img_tab_id = gr.State(0)
    txt2img_history = gr.State([])
    txt2img_history_index = gr.State(0)

    img2img_tab_id = gr.State(1)
    img2img_history = gr.State([])
    img2img_history_index = gr.State(0)

    upscale_tab_id = gr.State(2)
    # print(demo)
    # demo.css = "#gallery button:first-of-type {border: 2px solid blue}"
    with gr.Tabs() as tabs:
        with gr.TabItem("TextToImage", id=txt2img_tab_id.value) as txt2img_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    txt2img_prompt = gr.Textbox(value="Beautiful painting of a ghastly skeleton king in armor emerging from a medieval crypt by Frank Frazetta. Foreboding mood. uhd, high resolution, fine details, intricate details, centered, high quality, fine details, masterpiece, fine brushstrokes, dynamic movement", interactive=True, label="Prompt")
                    txt2img_negative_prompt = gr.Textbox(value="low quality, mid quality, low resolution, jpeg artifacts, watermark, text, poorly drawn, amateur", interactive=True, label="Negative Prompt")
                    txt2img_decoder_guidance_scale = gr.Slider(1,10,step=0.5, value=8.0, interactive=True, label="Guidance Scale For Image Generation")
                    with gr.Accordion('More Options', open=False):
                        txt2img_num_images_per_prompt = gr.Slider(1,8,step=1,value=2, label="Number of Images", visible=True, interactive=True)
                        txt2img_prior_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Prior Inference Steps", visible=True, interactive=True)
                        txt2img_decoder_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Decoder Inference Steps", visible=True, interactive=True)
                        txt2img_super_res_num_inference_steps = gr.Slider(1,20,step=1,value=7, label="Superres Inference Steps", visible=True, interactive=True)
                        txt2img_prior_guidance_scale = gr.Slider(1,10,step=0.5, value=4.0, label="Prior Guidance Scale", visible=True, interactive=True)
                        txt2img_manual_seed = gr.Number(value=-1, interactive=True, label="Seed")
                    with gr.Row():
                        txt2img_button = gr.Button(value="Generate Images from Text", variant="primary")
                with gr.Column(scale=1):
                    txt2img_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery").style(grid=[2], height="auto", elem_id="main-gallery")
                    with gr.Row():
                        txt2img_back_in_history_button = gr.Button(value="Back in Image History", variant="secondary")
                        txt2img_forward_in_history_button = gr.Button(value="Forward in Image History", variant="secondary")
                    with gr.Row():
                        txt2img_selected_image_index = gr.Number(value=1, interactive=True, label="Selected Image Index", precision=0)
                        # txt2img_selected_image_index = gr.Slider(value=1, step=1, minimum=1, maximum=1, interactive=True, label="Selected Image Index")
                        txt2img_send_to_img2img_button = gr.Button(value="Send to Image to Image", variant="secondary")
                        txt2img_send_to_upscale_button = gr.Button(value="Send to Upscale", variant="secondary")
        with gr.TabItem("ImageToImage", id=img2img_tab_id.value) as img2img_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    img2img_target_image = gr.Image(type="pil")
                    img2img_prompt = gr.Textbox(value="", interactive=True, label="Prompt")
                    img2img_negative_prompt = gr.Textbox(value="", interactive=True, label="Negative Prompt")
                    img2img_decoder_guidance_scale = gr.Slider(1,10,step=0.5, value=8.0, interactive=True, label="Guidance Scale For Image Generation")
                    with gr.Accordion('More Options', open=False):
                        img2img_num_images_per_prompt = gr.Slider(1,8,step=1,value=2, label="Number of Images", visible=True, interactive=False)
                        img2img_prior_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Prior Inference Steps", visible=True, interactive=False)
                        img2img_decoder_num_inference_steps = gr.Slider(10,100,step=5,value=25, label="Decoder Inference Steps", visible=True, interactive=True)
                        img2img_super_res_num_inference_steps = gr.Slider(1,20,step=1,value=7, label="Superres Inference Steps", visible=True, interactive=False)
                        img2img_prior_guidance_scale = gr.Slider(1,10,step=0.5, value=4.0, label="Prior Guidance Scale", visible=True, interactive=False)
                        img2img_manual_seed = gr.Number(value=-1, interactive=True, label="Seed")
                    with gr.Row():
                        img2img_button = gr.Button(value="Generate Images from Image", variant="primary")
                with gr.Column(scale=1):
                    img2img_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery").style(grid=[2], height="auto", elem_id="main-gallery")
                    with gr.Row():
                        img2img_back_in_history_button = gr.Button(value="Back in Image History", variant="secondary")
                        img2img_back_in_history_button.click(do_step_back_in_history, inputs=[img2img_history, img2img_history_index], outputs=[img2img_gallery, img2img_history, img2img_history_index], queue=False)

                        img2img_forward_in_history_button = gr.Button(value="Forward in Image History", variant="secondary")
                        img2img_forward_in_history_button.click(do_step_forward_in_history, inputs=[img2img_history, img2img_history_index], outputs=[img2img_gallery, img2img_history, img2img_history_index], queue=False)
                    with gr.Row():
                        img2img_selected_image_index = gr.Number(value=1, interactive=True, label="Selected Image Index", precision=0)
                        img2img_send_to_img2img_button = gr.Button(value="Send to Image to Image", variant="secondary")
                        img2img_send_to_upscale_button = gr.Button(value="Send to Upscale", variant="secondary")
        with gr.TabItem("Upscale", id=upscale_tab_id.value) as upscale_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    upscale_target_image = gr.Image(type="pil")
                    upscale_downsample_size = gr.Slider(value=256, minimum=64, maximum=256, step=32, interactive=True)
                    upscale_num_inference_steps = gr.Slider(value=100, minimum=5, maximum=100, step=5, interactive=True)
                    upscale_button = gr.Button(value="Upscale", variant="primary")
                with gr.Column(scale=1):
                    upscaled_image = gr.Image(type="pil", elem_id="upscaled-image").style(width=768, height=768)
                    upscale_save_button = gr.Button(value="Save Upscaled Image", variant="secondary")


        # BINDINGS                                                  
        # txt2img
        # txt2img history
        txt2img_back_in_history_button.click(do_step_back_in_history, inputs=[txt2img_history, txt2img_history_index], outputs=[txt2img_gallery, txt2img_history, txt2img_history_index], queue=False)
        txt2img_forward_in_history_button.click(do_step_forward_in_history, inputs=[txt2img_history, txt2img_history_index], outputs=[txt2img_gallery, txt2img_history, txt2img_history_index], queue=False)
        # txt2img generation
        txt2img_button.click(do_generate_txt2img, inputs=[txt2img_prompt, txt2img_negative_prompt, txt2img_num_images_per_prompt, txt2img_prior_num_inference_steps,txt2img_decoder_num_inference_steps, txt2img_super_res_num_inference_steps, txt2img_prior_guidance_scale, txt2img_decoder_guidance_scale, txt2img_manual_seed, txt2img_history, txt2img_history_index], outputs=[txt2img_gallery, txt2img_history, txt2img_history_index])

        # txt2img selection highlight
        # txt2img_selected_image_index.change(highlight_selected_image, inputs=[ txt2img_selected_image_index], outputs=[demo])

        # txt2img send buttons
        txt2img_send_to_img2img_button.click(do_send_image_from_history_with_prompts, inputs=[txt2img_history, txt2img_history_index, txt2img_selected_image_index, img2img_tab_id, txt2img_prompt, txt2img_negative_prompt], outputs=[img2img_target_image, tabs, img2img_prompt, img2img_negative_prompt])
        txt2img_send_to_upscale_button.click(do_send_image_from_history, inputs=[txt2img_history, txt2img_history_index, txt2img_selected_image_index, upscale_tab_id], outputs=[upscale_target_image, tabs])

        # img2img
        # img2img history
        img2img_back_in_history_button.click(do_step_back_in_history, inputs=[img2img_history, img2img_history_index], outputs=[img2img_gallery, img2img_history, img2img_history_index], queue=False)
        img2img_forward_in_history_button.click(do_step_forward_in_history, inputs=[img2img_history, img2img_history_index], outputs=[img2img_gallery, img2img_history, img2img_history_index], queue=False)
        # img2img generation
        img2img_button.click(do_generate_img2img, inputs=[img2img_target_image, img2img_prompt, img2img_negative_prompt, img2img_num_images_per_prompt,img2img_decoder_num_inference_steps, img2img_decoder_guidance_scale, img2img_manual_seed, img2img_history, img2img_history_index], outputs=[img2img_gallery, img2img_history, img2img_history_index])
        # img2img send buttons
        img2img_send_to_img2img_button.click(do_send_image_from_history, inputs=[img2img_history, img2img_history_index, img2img_selected_image_index, img2img_tab_id], outputs=[img2img_target_image, tabs])
        img2img_send_to_upscale_button.click(do_send_image_from_history, inputs=[img2img_history, img2img_history_index, img2img_selected_image_index, upscale_tab_id], outputs=[upscale_target_image, tabs])
        
        # upscale
        upscale_button.click(do_upscale_image, inputs=[upscale_target_image, upscale_downsample_size, upscale_num_inference_steps], outputs=[upscaled_image])
        upscale_save_button.click(do_save_image, inputs=[upscaled_image], outputs=[])
        
        # tabs
        txt2img_tab.select(do_update_selected_tab_index, inputs=[txt2img_tab_id], outputs=[tabs])
        img2img_tab.select(do_update_selected_tab_index, inputs=[img2img_tab_id], outputs=[tabs])
        upscale_tab.select(do_update_selected_tab_index, inputs=[upscale_tab_id], outputs=[tabs])
            
# demo.queue(concurrency_count=2)
# demo.launch(share=True)

demo.launch(share=False)