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
# import matplotlib.pyplot as plt

# def set_seed(seed):
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
#     down_blocks = OrderedDict()
#     mid_block = OrderedDict()
#     up_blocks = OrderedDict()

#     for key, value in layers.items():
#         if 'down_blocks' in key:
#             down_blocks[key] = value
#         elif 'mid_block' in key:
#             mid_block[key] = value
#         elif 'up_blocks' in key:
#             up_blocks[key] = value

#     sorted_layers = OrderedDict()
#     sorted_layers.update(down_blocks)
#     sorted_layers.update(mid_block)
#     sorted_layers.update(up_blocks)

#     return sorted_layers

# def get_empty_embeddings(encoder_hidden_states):
#     return torch.zeros_like(encoder_hidden_states)

# def modify_pipeline_for_cumulative_cross_attention(pipe):
#     layers = identify_layers(pipe)
#     sorted_layers = sort_layers(layers)
#     layer_names = list(sorted_layers.keys())
#     hooks = []

#     def forward_hook(module, input, output):
#         layer_index = layer_names.index(module.layer_name)
#         if layer_index >= module.target_layer_index:
#             return get_empty_embeddings(output)
#         return output

#     for layer, sublayers in sorted_layers.items():
#         for sublayer in sublayers:
#             module = pipe.unet.get_submodule(sublayer)
#             module.layer_name = layer
#             module.target_layer_index = 0
#             hook = module.register_forward_hook(forward_hook)
#             hooks.append(hook)

#     def generate_image_for_cumulative_layers(target_layer_index, prompt, seed, num_inference_steps, guidance_scale):
#         for layer in sorted_layers.values():
#             for sublayer in layer:
#                 module = pipe.unet.get_submodule(sublayer)
#                 module.target_layer_index = target_layer_index

#         with torch.no_grad():
#             generator = torch.Generator(device=pipe.device).manual_seed(seed)
#             image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
#         return image

#     return generate_image_for_cumulative_layers, sorted_layers, hooks, len(layer_names)

# def visualize_attention(pipe, prompt, layer_index, num_inference_steps, guidance_scale, generator, output_dir):
#     attention_store = []
    
#     def attention_callback(pipe, step_index, timestep, callback_kwargs):
#         if "cross_attention_kwargs" in callback_kwargs:
#             attn = callback_kwargs["cross_attention_kwargs"].get("attn")
#             if attn is not None:
#                 attention_store.append(attn.cpu())
#         return callback_kwargs

#     image = pipe(
#         prompt,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         generator=generator,
#         callback_on_step_end=attention_callback,
#         output_type="np"
#     ).images[0]
    
#     if attention_store:
#         attn = attention_store[-1]
#         attn_image = attn.sum(1).mean(0)
#         attn_image = (attn_image - attn_image.min()) / (attn_image.max() - attn_image.min())
        
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title("Generated Image")
#         plt.axis('off')
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(attn_image, cmap='viridis')
#         plt.title(f"Attention Map (Layer {layer_index})")
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"attention_layer_{layer_index}.png"))
#         plt.close()
#     else:
#         logging.warning(f"No attention data available for layer {layer_index}")

# def amplify_differences(image1, image2, amplification_factor=5):
#     diff = np.array(image1).astype(float) - np.array(image2).astype(float)
#     amplified_diff = diff * amplification_factor
    
#     pure_diff_image = np.clip(amplified_diff + 128, 0, 255).astype(np.uint8)
#     amplified_image = np.clip(np.array(image2).astype(float) + amplified_diff, 0, 255).astype(np.uint8)
    
#     return Image.fromarray(amplified_image), Image.fromarray(pure_diff_image)

# def create_comparison_image(images, output_dir, filename="layer_comparison.png"):
#     widths, heights = zip(*(i.size for i in images))
#     max_width = max(widths)
#     total_height = sum(heights)

#     comparison = Image.new('RGB', (max_width, total_height))
#     y_offset = 0
#     for img in images:
#         comparison.paste(img, (0, y_offset))
#         y_offset += img.size[1]

#     comparison_path = os.path.join(output_dir, filename)
#     comparison.save(comparison_path)
#     logging.info(f"Saved comparison image to {comparison_path}")
#     return comparison

# def main():
#     parser = argparse.ArgumentParser(description='逐层生成图像并比较交叉注意力层的影响。')
#     parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
#     parser.add_argument('--prompt', type=str, default='A black cat nestled on a table, abstract style', help='用于图像生成的提示词')
#     parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
#     parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
#     parser.add_argument('--output_dir', type=str, default='layer_outputs', help='保存输出图像的目录')
#     parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
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

#     generate_image_for_cumulative_layers, sorted_layers, hooks, total_layers = modify_pipeline_for_cumulative_cross_attention(pipe)

#     os.makedirs(args.output_dir, exist_ok=True)

#     layer_images = []
#     amplified_images = []
#     pure_diff_images = []
#     generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

#     logging.info("Generating base image with empty condition...")
#     base_image = generate_image_for_cumulative_layers(0, "", args.seed, args.num_inference_steps, args.guidance_scale)
#     base_image_path = os.path.join(args.output_dir, "base_image.png")
#     base_image.save(base_image_path)
#     previous_image = base_image

#     logging.info("Starting image generation for each layer...")
#     for i in range(total_layers + 1):  # +1 to include the all layers image
#         try:
#             image = generate_image_for_cumulative_layers(i, args.prompt, args.seed, args.num_inference_steps, args.guidance_scale)
#             if i < total_layers:
#                 layer_name = list(sorted_layers.keys())[i]
#                 safe_layer_name = f"cumulative_{i}_{layer_name.replace('.', '_')}"
#             else:
#                 safe_layer_name = "all_layers"
            
#             image_path = os.path.join(args.output_dir, f"{safe_layer_name}.png")
#             image.save(image_path)
            
#             amplified_image, pure_diff_image = amplify_differences(image, previous_image)
#             amplified_image_path = os.path.join(args.output_dir, f"amplified_{safe_layer_name}.png")
#             amplified_image.save(amplified_image_path)
#             amplified_images.append(amplified_image)
            
#             pure_diff_image_path = os.path.join(args.output_dir, f"pure_diff_{safe_layer_name}.png")
#             pure_diff_image.save(pure_diff_image_path)
#             pure_diff_images.append(pure_diff_image)
            
#             layer_images.append(image)
#             previous_image = image  # 更新前一层的图像
            
#             visualize_attention(pipe, args.prompt, i, args.num_inference_steps, args.guidance_scale, generator, args.output_dir)
            
#             logging.info(f"Generated image for {'all layers' if i == total_layers else f'cumulative layers up to {i}'}")
            
#         except Exception as e:
#             logging.error(f"Error processing layer {i}: {e}")

#     try:
#         create_comparison_image(layer_images, args.output_dir, "layer_comparison.png")
#         create_comparison_image(amplified_images, args.output_dir, "amplified_comparison.png")
#         create_comparison_image(pure_diff_images, args.output_dir, "pure_diff_comparison.png")
#     except Exception as e:
#         logging.error(f"Error creating comparison images: {e}")

#     logging.info("Process completed. Check the output directory for results.")

#     for hook in hooks:
#         hook.remove()

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
import matplotlib.pyplot as plt
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
    down_blocks = OrderedDict()
    mid_block = OrderedDict()
    up_blocks = OrderedDict()

    for key, value in layers.items():
        if 'down_blocks' in key:
            down_blocks[key] = value
        elif 'mid_block' in key:
            mid_block[key] = value
        elif 'up_blocks' in key:
            up_blocks[key] = value

    sorted_layers = OrderedDict()
    sorted_layers.update(down_blocks)
    sorted_layers.update(mid_block)
    sorted_layers.update(up_blocks)

    return sorted_layers

def get_empty_embeddings(encoder_hidden_states):
    return torch.zeros_like(encoder_hidden_states)

def modify_pipeline_for_cumulative_cross_attention(pipe):
    layers = identify_layers(pipe)
    sorted_layers = sort_layers(layers)
    
    total_layers = len(sorted_layers)
    logging.info(f"Total layers: {total_layers}")
    for layer, sublayers in sorted_layers.items():
        logging.info(f"Layer: {layer}")
        for sublayer in sublayers:
            logging.info(f"  Sublayer: {sublayer}")
    
    layer_names = list(sorted_layers.keys())
    hooks = []

    def forward_hook(module, input, output):
        layer_index = layer_names.index(module.layer_name)
        if layer_index >= module.target_layer_index:
            return get_empty_embeddings(output)
        return output

    for layer, sublayers in sorted_layers.items():
        for sublayer in sublayers:
            module = pipe.unet.get_submodule(sublayer)
            module.layer_name = layer
            module.target_layer_index = 0
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    def generate_image_for_cumulative_layers(target_layer_index, prompt, seed, num_inference_steps, guidance_scale):
        for layer in sorted_layers.values():
            for sublayer in layer:
                module = pipe.unet.get_submodule(sublayer)
                module.target_layer_index = target_layer_index

        with torch.no_grad():
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        return image

    return generate_image_for_cumulative_layers, sorted_layers, hooks, len(layer_names)

def amplify_differences(image1, image2, amplification_factor=5):
    diff = np.array(image1).astype(float) - np.array(image2).astype(float)
    amplified_diff = diff * amplification_factor
    
    pure_diff_image = np.clip(amplified_diff + 128, 0, 255).astype(np.uint8)
    amplified_image = np.clip(np.array(image2).astype(float) + amplified_diff, 0, 255).astype(np.uint8)
    
    return Image.fromarray(amplified_image), Image.fromarray(pure_diff_image)
def create_difference_heatmap(image1, image2, amplification_factor=1):
    # 转换为numpy数组并计算差异
    img1 = np.array(image1).astype(float)
    img2 = np.array(image2).astype(float)
    diff = np.abs(img1 - img2)  # 使用绝对差值
    
    # 如果是彩色图片,将差异转换为单通道
    if len(diff.shape) == 3:
        diff = np.mean(diff, axis=2)
    
    # 归一化差异到[0, 1]范围
    diff_normalized = diff / diff.max() if diff.max() > 0 else diff
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_normalized, cmap='hot')  # 使用hot色图
    plt.colorbar(label='Difference')
    plt.title('Difference Heatmap')
    
    # 保存热力图
    plt.savefig('difference_heatmap.png')
    plt.close()
    
    # 返回热力图数组
    return diff_normalized
# def amplify_differences(image1, image2, amplification_factor=5):
#     # 将图像转换为numpy数组并转为浮点型
#     img1 = np.array(image1).astype(float)
#     img2 = np.array(image2).astype(float)
    
#     # 计算差异
#     diff = img1 - img2
    
#     # 归一化差异到[-1, 1]范围
#     max_abs_diff = np.max(np.abs(diff))
#     if max_abs_diff > 0:
#         normalized_diff = diff / max_abs_diff  # 归一化操作
#     else:
#         normalized_diff = diff
    
#     # 放大归一化后的差异
#     amplified_diff = normalized_diff * amplification_factor
    
#     # 生成差异图像：将[-1, 1]映射到[0, 255]
#     # (x + 1) * 127.5 将[-1, 1]映射到[0, 255]
#     pure_diff_image = ((amplified_diff + 1) * 127.5).astype(np.uint8)
    
#     # 生成放大后的图像
#     # 将归一化并放大的差异重新缩放回原始比例
#     scaled_diff = amplified_diff * max_abs_diff
#     amplified_image = np.clip(img2 + scaled_diff, 0, 255).astype(np.uint8)
    
#     return Image.fromarray(amplified_image), Image.fromarray(pure_diff_image)
def create_comparison_image(images, output_dir, filename="layer_comparison.png"):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    comparison = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    comparison_path = os.path.join(output_dir, filename)
    comparison.save(comparison_path)
    logging.info(f"Saved comparison image to {comparison_path}")
    return comparison
def main():
    parser = argparse.ArgumentParser(description='逐层生成图像并比较交叉注意力层的影响。')
    parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
    parser.add_argument('--prompt', type=str, default='A black cat nestled on a table, abstract style', help='用于图像生成的提示词')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
    parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
    parser.add_argument('--output_dir', type=str, default='layer_outputs', help='保存输出图像的目录')
    parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
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

    generate_image_for_cumulative_layers, sorted_layers, hooks, total_layers = modify_pipeline_for_cumulative_cross_attention(pipe)

    os.makedirs(args.output_dir, exist_ok=True)

    layer_images = []
    amplified_images = []
    pure_diff_images = []

    logging.info("Generating base image with empty condition...")
    base_image = generate_image_for_cumulative_layers(0, "", args.seed, args.num_inference_steps, args.guidance_scale)
    base_image_path = os.path.join(args.output_dir, "base_image.png")
    base_image.save(base_image_path)
    layer_images = [base_image]  # 将基准图像添加到层图像列表
    previous_image = base_image

    logging.info("Starting image generation for each layer...")
    for i in range(1, total_layers + 1):  # 从1开始，到total_layers（包括）
        try:
            image = generate_image_for_cumulative_layers(i, args.prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            if i < total_layers:
                layer_name = list(sorted_layers.keys())[i-1]  # 因为我们从1开始，所以这里需要-1
                safe_layer_name = f"cumulative_{i}_{layer_name.replace('.', '_')}"
            else:
                safe_layer_name = "all_layers"
            
            image_path = os.path.join(args.output_dir, f"{safe_layer_name}.png")
            image.save(image_path)
            
            amplified_image, pure_diff_image = amplify_differences(image, previous_image)
            diff_normalized=create_difference_heatmap(image, previous_image, amplification_factor=5)
            amplified_image_path = os.path.join(args.output_dir, f"amplified_{safe_layer_name}.png")
            amplified_image.save(amplified_image_path)
            amplified_images.append(amplified_image)
            
            pure_diff_image_path = os.path.join(args.output_dir, f"pure_diff_{safe_layer_name}.png")
            pure_diff_image.save(pure_diff_image_path)
            pure_diff_images.append(pure_diff_image)
            
            layer_images.append(image)
            previous_image = image  # 更新前一层的图像
            
            logging.info(f"Generated image for {'all layers' if i == total_layers else f'cumulative layers up to {i}'}")
            
        except Exception as e:
            logging.error(f"Error processing layer {i}: {e}")

    try:
        create_comparison_image(layer_images, args.output_dir, "layer_comparison.png")
        create_comparison_image(amplified_images, args.output_dir, "amplified_comparison.png")
        create_comparison_image(pure_diff_images, args.output_dir, "pure_diff_comparison.png")
    except Exception as e:
        logging.error(f"Error creating comparison images: {e}")

    logging.info("Process completed. Check the output directory for results.")

    for hook in hooks:
        hook.remove()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
