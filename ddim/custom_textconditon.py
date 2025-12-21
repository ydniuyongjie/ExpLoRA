import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
import gc
from types import MethodType
from collections import defaultdict
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def identify_layers(pipe):
    layers = defaultdict(list)
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and hasattr(module, 'processor'):
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'attentions':
                    layer_key = '.'.join(parts[:i+2])  # Include the number after 'attentions'
                    layers[layer_key].append(name)
                    break
    return dict(layers)

def create_cross_attention_wrapper(original_forward):
    def wrapper(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif not self.is_target_layer:
            encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    return wrapper

def modify_pipeline_for_layer_specific_cross_attention(pipe):
    layers = identify_layers(pipe)
    total_layers = len(layers)
    print(f"Total layers: {total_layers}")
    for layer, sublayers in layers.items():
        print(f"Layer: {layer}")
        for sublayer in sublayers:
            print(f"  Sublayer: {sublayer}")

    original_forward_functions = {}

    for layer, sublayers in layers.items():
        for sublayer in sublayers:
            module = pipe.unet.get_submodule(sublayer)
            original_forward_functions[sublayer] = module.forward
            module.forward = MethodType(create_cross_attention_wrapper(module.forward), module)
            module.is_target_layer = False

    def generate_image_for_specific_layer(target_layer, prompt):
        print(f"Generating image for layer: {target_layer}")
        
        for layer, sublayers in layers.items():
            for sublayer in sublayers:
                module = pipe.unet.get_submodule(sublayer)
                module.is_target_layer = (layer == target_layer)

        with torch.no_grad():
            generator = torch.Generator(device=pipe.device).manual_seed(469)
            image = pipe(prompt, num_inference_steps=50, guidance_scale=10, generator=generator).images[0]

        return image

    return generate_image_for_specific_layer, layers, original_forward_functions

def create_comparison_image(images):
    width, height = images[0].size
    comparison = Image.new('RGB', (width * len(images), height))
    for i, img in enumerate(images):
        comparison.paste(img, (i * width, 0))
    return comparison

def main():
    set_seed(469)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    generate_image_for_specific_layer, layers, original_forward_functions = modify_pipeline_for_layer_specific_cross_attention(pipe)
#——————————————————————————————————————————————————————————————————————————————————————
    # prompt = "A white cat nestling on a table, Van Gogh style"
    prompt ="A black cat nestled on a table, abstract style"
#——————————————————————————————————————————————————————————————————————————————————————
    os.makedirs("layer_outputs", exist_ok=True)

    layer_images = []
    for layer_name in layers.keys():
        try:
            image = generate_image_for_specific_layer(layer_name, prompt)
            # 使用层名称作为文件名，但要替换可能在文件名中无效的字符
            safe_layer_name = layer_name.replace('.', '_')
            image.save(f"layer_outputs/{safe_layer_name}.png")
            layer_images.append(image)
            print(f"Generated image for layer {layer_name}")
        except Exception as e:
            print(f"Error generating image for layer {layer_name}: {e}")

    # Generate image with all layers active
    try:
        for sublayer, original_forward in original_forward_functions.items():
            module = pipe.unet.get_submodule(sublayer)
            module.forward = original_forward

        generator = torch.Generator(device=pipe.device).manual_seed(469)
        all_layers_image = pipe(prompt, num_inference_steps=50, guidance_scale=10, generator=generator).images[0]
        all_layers_image.save("layer_outputs/all_layers.png")
        layer_images.append(all_layers_image)
        print("Generated image using all layers")
    except Exception as e:
        print(f"Error generating image for all layers: {e}")

    try:
        comparison = create_comparison_image(layer_images)
        comparison.save("layer_comparison.png")
        print("Created comparison image")
    except Exception as e:
        print(f"Error creating comparison image: {e}")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
