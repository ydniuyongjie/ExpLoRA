# import torch
# from diffusers import StableDiffusionXLPipeline
# from PIL import Image
# import os
# import gc
# from collections import defaultdict, OrderedDict
# import random
# import numpy as np
# import argparse
# import logging
# import json

# def set_seed(seed):
#     """设置所有随机数生成器的种子以确保结果可重现。"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def identify_layers(pipe):
#     layers = defaultdict(list)
#     for name, module in pipe.unet.named_modules():
#         if 'attn2' in name and hasattr(module, 'processor'):
#             parts = name.split('.')
#             for i, part in enumerate(parts):
#                 if part == 'attentions':
#                     layer_key = '.'.join(parts[:i+2])
#                     layers[layer_key].append(name)
#                     break
#     return dict(layers)

# def sort_layers(layers):
#     def sort_key(item):
#         key = item[0]
#         parts = key.split('.')
#         if parts[0] == 'mid_block':
#             return (1, 0, 0)
#         elif parts[0] == 'down_blocks':
#             return (0, int(parts[1]), int(parts[3]))
#         elif parts[0] == 'up_blocks':
#             return (2, int(parts[1]), int(parts[3]))
#         else:
#             return (3, 0, 0)  # 为未知类型提供默认排序

#     return OrderedDict(sorted(layers.items(), key=sort_key))

# def load_prompts(json_file):
#     """从JSON文件加载提示词"""
#     with open(json_file, 'r') as f:
#         return json.load(f)

# def modify_pipeline_for_layer_specific_cross_attention(pipe, prompts):
#     """修改 pipeline 以允许特定层的交叉注意力。"""
#     layers = identify_layers(pipe)
#     sorted_layers = sort_layers(layers)
#     total_layers = len(sorted_layers)
#     logging.info(f"Total layers: {total_layers}")
#     for i, (layer, sublayers) in enumerate(sorted_layers.items(), 1):
#         logging.info(f"Layer {i}: {layer}")
#         # 移除了对 sublayers 的输出

#     hooks = []

#     def create_forward_hook(layer_name):
#         def forward_hook(module, input, output):
#             if module.is_target_layer:
#                 return output
#             return torch.zeros_like(output)
#         return forward_hook

#     for layer, sublayers in layers.items():
#         for sublayer in sublayers:
#             module = pipe.unet.get_submodule(sublayer)
#             module.is_target_layer = False
#             hook = module.register_forward_hook(create_forward_hook(layer))
#             hooks.append(hook)

#     def generate_image_for_specific_layer(target_layer, prompt, seed, num_inference_steps, guidance_scale):
#         """为特定层生成图像。"""
#         logging.info(f"Generating image for layer: {target_layer}")

#         for layer, sublayers in layers.items():
#             for sublayer in sublayers:
#                 module = pipe.unet.get_submodule(sublayer)
#                 module.is_target_layer = (layer == target_layer)

#         with torch.no_grad():
#             generator = torch.Generator(device=pipe.device).manual_seed(seed)
#             image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]

#         return image

#     return generate_image_for_specific_layer, layers, hooks

# def create_comparison_image(images, output_dir, layer_order):
#     """创建一个包含所有生成图像的比较图像，按照层的生成顺序排列，合成图放在最后。"""
#     # 只使用成功生成的图像
#     available_images = {layer: img for layer, img in images.items() if img is not None}
    
#     if 'all_layers' not in available_images:
#         logging.error("'all_layers' image is missing")
#         return None

#     ordered_images = [available_images.get(layer) for layer in layer_order if layer in available_images]
#     ordered_images = [img for img in ordered_images if img is not None]  # 移除None值
#     ordered_images.append(available_images['all_layers'])
    
#     if not ordered_images:
#         logging.error("No images available for comparison")
#         return None

#     widths, heights = zip(*(i.size for i in ordered_images))
#     total_width = sum(widths)
#     max_height = max(heights)

#     comparison = Image.new('RGB', (total_width, max_height))
    
#     x_offset = 0
#     for img in ordered_images:
#         comparison.paste(img, (x_offset, 0))
#         x_offset += img.size[0]

#     comparison_dir = os.path.join(output_dir, "comparison")
#     os.makedirs(comparison_dir, exist_ok=True)
#     comparison_path = os.path.join(comparison_dir, "layer_comparison.png")
#     comparison.save(comparison_path)
#     logging.info(f"Saved comparison image to {comparison_path}")

#     return comparison

# def main():
#     """主函数，运行整个图像生成和比较过程。"""
#     parser = argparse.ArgumentParser(description='逐层生成图像并比较交叉注意力层的影响。')
#     parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
#     parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
#     parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
#     parser.add_argument('--output_dir', type=str, default='layer_outputs', help='保存输出图像的目录')
#     parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
#     parser.add_argument('--prompts_file', type=str, default='prompts.json', help='包含层特定提示词的JSON文件')
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

#     set_seed(args.seed)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logging.info(f"Using device: {device}")

#     try:
#         pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True)
#         pipe = pipe.to(device)
#     except Exception as e:
#         logging.error(f"Error loading the model: {e}")
#         return

#     prompts = load_prompts(args.prompts_file)
#     generate_image_for_specific_layer, layers, hooks = modify_pipeline_for_layer_specific_cross_attention(pipe, prompts)

#     os.makedirs(args.output_dir, exist_ok=True)

#     sorted_layers = sort_layers(layers)
#     layer_order = list(sorted_layers.keys())

#     # Generate image for all layers first
#     try:
#         generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
#         all_layers_image = pipe(prompts['all_layers'], num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
#         all_layers_image.save(os.path.join(args.output_dir, "all_layers.png"))
#         logging.info("Generated image using all layers")
#     except Exception as e:
#         logging.error(f"Error generating image for all layers: {e}")

#     layer_images = {}
#     for i, layer_name in enumerate(layer_order, 1):
#         prompt_key = f"layer_{i}"
#         prompt = prompts.get(prompt_key, "")
#         if not prompt:
#             logging.info(f"Skipping layer {layer_name} due to empty prompt")
#             continue

#         try:
#             image = generate_image_for_specific_layer(layer_name, prompt, args.seed, args.num_inference_steps, args.guidance_scale)
#             safe_layer_name = layer_name.replace('.', '_')
#             image_path = os.path.join(args.output_dir, f"{safe_layer_name}.png")
#             image.save(image_path)
#             layer_images[layer_name] = image
#             logging.info(f"Generated image for layer {layer_name}")
#         except Exception as e:
#             logging.error(f"Error generating image for layer {layer_name}: {e}")
#             layer_images[layer_name] = None  # 将失败的层设置为None

#     for hook in hooks:
#         hook.remove()

#     # 生成 all_layers 图像
#     try:
#         generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
#         all_layers_image = pipe(prompts['all_layers'], num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
#         all_layers_image.save(os.path.join(args.output_dir, "all_layers.png"))
#         layer_images['all_layers'] = all_layers_image
#         logging.info("Generated image using all layers")
#     except Exception as e:
#         logging.error(f"Error generating image for all layers: {e}")
#         layer_images['all_layers'] = None

#     try:
#         comparison_image = create_comparison_image(layer_images, args.output_dir, layer_order)
#         if comparison_image is None:
#             logging.error("Failed to create comparison image")
#     except Exception as e:
#         logging.error(f"Error creating comparison image: {e}")

#     del pipe
#     torch.cuda.empty_cache()
#     gc.collect()

# if __name__ == "__main__":
#     main()

# import torch
# from diffusers import StableDiffusionXLPipeline
# from PIL import Image
# import os
# import gc
# from collections import defaultdict, OrderedDict
# import random
# import numpy as np
# import argparse
# import logging
# import json

# def set_seed(seed):
#     """设置所有随机数生成器的种子以确保结果可重现。"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def identify_layers(pipe):
#     layers = defaultdict(list)
#     for name, module in pipe.unet.named_modules():
#         if 'attn2' in name and hasattr(module, 'processor'):
#             parts = name.split('.')
#             for i, part in enumerate(parts):
#                 if part == 'attentions':
#                     layer_key = '.'.join(parts[:i+2])
#                     layers[layer_key].append(name)
#                     break
#     return dict(layers)

# def sort_layers(layers):
#     def sort_key(item):
#         key = item[0]
#         parts = key.split('.')
#         if parts[0] == 'mid_block':
#             return (1, 0, 0)
#         elif parts[0] == 'down_blocks':
#             return (0, int(parts[1]), int(parts[3]))
#         elif parts[0] == 'up_blocks':
#             return (2, int(parts[1]), int(parts[3]))
#         else:
#             return (3, 0, 0)  # 为未知类型提供默认排序

#     return OrderedDict(sorted(layers.items(), key=sort_key))

# def load_prompts(json_file):
#     """从JSON文件加载提示词"""
#     with open(json_file, 'r') as f:
#         return json.load(f)

# def get_empty_embeddings(encoder_hidden_states):
#     if encoder_hidden_states.dim() == 3:
#         return torch.zeros_like(encoder_hidden_states)
#     elif encoder_hidden_states.dim() == 4:
#         # 对于注意力权重，我们可能需要保持某些维度不变
#         return torch.zeros_like(encoder_hidden_states)
#     else:
#         raise ValueError(f"Unexpected dimension: {encoder_hidden_states.dim()}")

# def modify_pipeline_for_cumulative_cross_attention(pipe, prompts):
#     layers = identify_layers(pipe)
#     sorted_layers = sort_layers(layers)
    
#     total_layers = len(sorted_layers)
#     logging.info(f"Total layers: {total_layers}")
#     for layer, sublayers in sorted_layers.items():
#         logging.info(f"Layer: {layer}")
#         for sublayer in sublayers:
#             logging.info(f"  Sublayer: {sublayer}")
    
#     layer_names = list(sorted_layers.keys())
#     hooks = []

#     def forward_hook(module, input, output):
#         layer_index = layer_names.index(module.layer_name)
#         prompt = prompts.get(f"layer_{layer_index + 1}", "")
#         if prompt:
#             return output
#         else:
#             if isinstance(output, torch.Tensor):
#                 return get_empty_embeddings(output)
#             elif isinstance(output, tuple):
#                 return tuple(get_empty_embeddings(o) if isinstance(o, torch.Tensor) else o for o in output)
#             else:
#                 raise ValueError(f"Unexpected output type: {type(output)}")

#     for layer, sublayers in sorted_layers.items():
#         for sublayer in sublayers:
#             module = pipe.unet.get_submodule(sublayer)
#             module.layer_name = layer
#             hook = module.register_forward_hook(forward_hook)
#             hooks.append(hook)

#     def generate_cumulative_image(prompt, seed, num_inference_steps, guidance_scale):
#         with torch.no_grad():
#             generator = torch.Generator(device=pipe.device).manual_seed(seed)
#             image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
#         return image

#     return generate_cumulative_image, sorted_layers, hooks

# def main():
#     parser = argparse.ArgumentParser(description='生成一张累积了所有层条件的图像。')
#     parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
#     parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
#     parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
#     parser.add_argument('--output_dir', type=str, default='output', help='保存输出图像的目录')
#     parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
#     parser.add_argument('--prompts_file', type=str, default='prompts.json', help='包含层特定提示词的JSON文件')
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

#     set_seed(args.seed)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logging.info(f"Using device: {device}")

#     try:
#         pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True)
#         pipe = pipe.to(device)
#     except Exception as e:
#         logging.error(f"Error loading the model: {e}")
#         return

#     prompts = load_prompts(args.prompts_file)

#     os.makedirs(args.output_dir, exist_ok=True)

#     try:
#         generate_cumulative_image, sorted_layers, hooks = modify_pipeline_for_cumulative_cross_attention(pipe, prompts)

#         # 生成累积层的图像
#         all_layers_prompt = prompts.get("all_layers", "An image")
#         cumulative_image = generate_cumulative_image(all_layers_prompt, args.seed, args.num_inference_steps, args.guidance_scale)
#         cumulative_path = os.path.join(args.output_dir, "cumulative_image.png")
#         cumulative_image.save(cumulative_path)
#         logging.info(f"Saved cumulative image to {cumulative_path}")

#     except Exception as e:
#         logging.error(f"Error generating images: {e}")
#         logging.exception(e)
#     finally:
#         for hook in hooks:
#             hook.remove()

#     del pipe
#     torch.cuda.empty_cache()
#     gc.collect()

# if __name__ == "__main__":
#     main()


import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
import gc
from collections import defaultdict, OrderedDict
import random
import numpy as np
import argparse
import logging
import json

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
                    layer_key = '.'.join(parts[:i+2])
                    layers[layer_key].append(name)
                    break
    return dict(layers)

def sort_layers(layers):
    def sort_key(item):
        key = item[0]
        parts = key.split('.')
        if parts[0] == 'mid_block':
            return (1, 0, 0)
        elif parts[0] == 'down_blocks':
            return (0, int(parts[1]), int(parts[3]))
        elif parts[0] == 'up_blocks':
            return (2, int(parts[1]), int(parts[3]))
        else:
            return (3, 0, 0)
    return OrderedDict(sorted(layers.items(), key=sort_key))

def load_prompts(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def get_empty_embeddings(encoder_hidden_states):
    if encoder_hidden_states.dim() == 3:
        return torch.zeros_like(encoder_hidden_states)
    elif encoder_hidden_states.dim() == 4:
        return torch.zeros_like(encoder_hidden_states)
    else:
        raise ValueError(f"Unexpected dimension: {encoder_hidden_states.dim()}")

def modify_pipeline_for_layerwise_prompts(pipe, prompts):
    layers = identify_layers(pipe)
    sorted_layers = sort_layers(layers)
    
    total_layers = len(sorted_layers)
    logging.info(f"Total layers: {total_layers}")
    for layer in sorted_layers.keys():
        logging.info(f"Layer: {layer}")
    
    layer_names = list(sorted_layers.keys())
    hooks = []

    def forward_hook(module, input, output):
        layer_index = layer_names.index(module.layer_name)
        prompt = prompts.get(f"layer_{layer_index + 1}", "")
        
        if prompt:
            # 使用层级特定的提示词生成新的嵌入
            text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(pipe.device)
            prompt_embeds = pipe.text_encoder(text_inputs.input_ids)[0]
            return prompt_embeds
        else:
            return get_empty_embeddings(output)

    for layer, sublayers in sorted_layers.items():
        for sublayer in sublayers:
            module = pipe.unet.get_submodule(sublayer)
            module.layer_name = layer
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    def generate_layerwise_image(prompt, seed, num_inference_steps, guidance_scale):
        with torch.no_grad():
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        return image

    return generate_layerwise_image, sorted_layers, hooks

def modify_pipeline_for_cumulative_cross_attention(pipe, prompts):
    layers = identify_layers(pipe)
    sorted_layers = sort_layers(layers)
    
    total_layers = len(sorted_layers)
    logging.info(f"Total layers: {total_layers}")
    for layer in sorted_layers.keys():
        logging.info(f"Layer: {layer}")
    
    layer_names = list(sorted_layers.keys())
    hooks = []

    def forward_hook(module, input, output):
        layer_index = layer_names.index(module.layer_name)
        prompt = prompts.get(f"layer_{layer_index + 1}", "")
        if prompt:
            return output
        else:
            if isinstance(output, torch.Tensor):
                return get_empty_embeddings(output)
            elif isinstance(output, tuple):
                return tuple(get_empty_embeddings(o) if isinstance(o, torch.Tensor) else o for o in output)
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")

    for layer, sublayers in sorted_layers.items():
        for sublayer in sublayers:
            module = pipe.unet.get_submodule(sublayer)
            module.layer_name = layer
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    def generate_cumulative_image(prompt, seed, num_inference_steps, guidance_scale):
        with torch.no_grad():
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        return image

    return generate_cumulative_image, sorted_layers, hooks

def create_comparison_image(cumulative_image, all_layers_image):
    width, height = cumulative_image.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(cumulative_image, (0, 0))
    comparison.paste(all_layers_image, (width, 0))
    return comparison

def main():
    parser = argparse.ArgumentParser(description='生成一张累积了所有层条件的图像和一张使用所有层的图像，并创建比较图。')
    parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
    parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
    parser.add_argument('--output_dir', type=str, default='output', help='保存输出图像的目录')
    parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
    parser.add_argument('--prompts_file', type=str, default='prompts.json', help='包含层特定提示词的JSON文件')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, local_files_only=True)
        pipe = pipe.to(device)
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return

    prompts = load_prompts(args.prompts_file)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        generate_cumulative_image, sorted_layers, hooks = modify_pipeline_for_cumulative_cross_attention(pipe, prompts)

        # 生成累积层的图像
        all_layers_prompt = prompts.get("all_layers", "An image")
        cumulative_image = generate_cumulative_image(all_layers_prompt, args.seed, args.num_inference_steps, args.guidance_scale)
        cumulative_path = os.path.join(args.output_dir, "cumulative_image.png")
        cumulative_image.save(cumulative_path)
        logging.info(f"Saved cumulative image to {cumulative_path}")

        # 移除钩子以生成使用所有层的图像
        for hook in hooks:
            hook.remove()

        # 生成使用所有层的图像
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        all_layers_image = pipe(all_layers_prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
        all_layers_path = os.path.join(args.output_dir, "all_layers_image.png")
        all_layers_image.save(all_layers_path)
        logging.info(f"Saved all layers image to {all_layers_path}")

        # 创建比较图像
        comparison_image = create_comparison_image(cumulative_image, all_layers_image)
        comparison_path = os.path.join(args.output_dir, "comparison_image.png")
        comparison_image.save(comparison_path)
        logging.info(f"Saved comparison image to {comparison_path}")

    except Exception as e:
        logging.error(f"Error generating images: {e}")
        logging.exception(e)
    finally:
        # 确保所有钩子都被移除
        for hook in hooks:
            hook.remove()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()

