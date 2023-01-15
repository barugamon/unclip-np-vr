from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline, UnCLIPImageVariationPipeline
from diffusers import LDMSuperResolutionPipeline, StableDiffusionUpscalePipeline
from transformers import CLIPVisionModelWithProjection, CLIPFeatureExtractor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print('new_images', new_images)
    
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
    print('new_images', new_images)

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

with gr.Blocks() as demo:
    history = gr.State([])
    history_index = gr.State(0)
    # print(demo)
    # demo.css = "#gallery button:first-of-type {border: 2px solid blue}"
    with gr.Tab('Txt2Img'):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(value="Masterful painting of a mysterious pyramid in a lush jungle!!! by Rene Magritte. 4k, 8k, high quality, masterpiece, fine brushstrokes, surreal atmosphere.", interactive=True, label="Prompt")
                negative_prompt = gr.Textbox(value="low quality, mid quality, low resolution, jpeg artifacts, watermark, text, poorly drawn, amateur", interactive=True, label="Negative Prompt")
                num_images_per_prompt = gr.Slider(1,8,step=1,value=4, interactive=True, label="Number of Images", visible=False)
                prior_num_inference_steps = gr.Slider(10,100,step=5,value=25, interactive=True, label="Prior Inference Steps", visible=False)
                decoder_num_inference_steps = gr.Slider(10,100,step=5,value=25, interactive=True, label="Decoder Inference Steps", visible=False)
                super_res_num_inference_steps = gr.Slider(1,20,step=1,value=7, interactive=True, label="Superres Inference Steps", visible=False)
                prior_guidance_scale = gr.Slider(1,10,step=0.5, value=4.0, interactive=True, label="Prior Guidance Scale", visible=False)
                decoder_guidance_scale_ig = gr.Slider(1,10,step=0.5, value=4.0, interactive=True, label="Guidance Scale For Image Generation")
                manual_seed = gr.Number(value=-1, interactive=True, label="Seed")
                
                generate_button = gr.Button(value="Generate Images", variant="primary")
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery").style(grid=[2], height="auto", elem_id="main-gallery", container=True)
                
                with gr.Row():
                    back_in_history_button = gr.Button(value="Back in History", variant="secondary")
                    back_in_history_button.click(step_back_in_history, inputs=[history, history_index], outputs=[gallery, history, history_index])

                    forward_in_history_button = gr.Button(value="Forward in History", variant="secondary")
                    forward_in_history_button.click(step_forward_in_history, inputs=[history, history_index], outputs=[gallery, history, history_index])

                with gr.Row():
                    selected_image_index = gr.Radio([1,2,3,4], value=1, interactive=True, label="Selected Image for Variations")
                    decoder_guidance_scale_vg = gr.Slider(1,10,step=0.5, value=6.0, interactive=True, label="Guidance Scale For Variation Generation")

                # selected_image_index.change(highlight_selected_image, inputs=[demo, selected_image_index], outputs=[])

                variations_button = gr.Button(value=f"Generate Variations from Selected Image", variant="primary")
                variations_button.click(generate_variations, inputs=[gallery, prompt, negative_prompt, selected_image_index, num_images_per_prompt, decoder_num_inference_steps, decoder_guidance_scale_vg, history, history_index], outputs=[gallery, history, history_index])

                generate_button.click(generate_images, inputs=[prompt, negative_prompt, num_images_per_prompt, prior_num_inference_steps, decoder_num_inference_steps, super_res_num_inference_steps, prior_guidance_scale, decoder_guidance_scale_ig, manual_seed, history, history_index], outputs=[gallery, history, history_index])
                # generate_button.click

            
# for fast reload use "gradio playground.py"
demo.launch()