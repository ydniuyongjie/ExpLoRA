import torch
from diffusers import AutoencoderKL,StableDiffusionXLPipeline
from PIL import Image
import os
import gc
from collections import defaultdict, OrderedDict
import random
import numpy as np
import argparse
import logging

def set_seed(seed):
    """设置所有随机数生成器的种子以确保结果可重现。"""
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
            return (3, 0, 0)  # 为未知类型提供默认排序

    return OrderedDict(sorted(layers.items(), key=sort_key))

def get_empty_embeddings(encoder_hidden_states):
    """生成与给定嵌入相同形状的空嵌入（零张量）。"""
    return torch.zeros_like(encoder_hidden_states)

def modify_pipeline_for_layer_specific_cross_attention(pipe):
    """修改 pipeline 以允许特定层的交叉注意力。"""
    layers = identify_layers(pipe)
    sorted_layers = sort_layers(layers)
    total_layers = len(sorted_layers)
    logging.info(f"Total layers: {total_layers}")
    for layer, sublayers in layers.items():
        logging.info(f"Layer: {layer}")
    # for sublayer in sublayers:
    #     logging.info(f" Sublayer: {sublayer}")

    hooks = []

    def forward_hook(module, input, output):
        if not module.is_target_layer:
            return get_empty_embeddings(output)
        return output

    for layer, sublayers in layers.items():
        for sublayer in sublayers:
            module = pipe.unet.get_submodule(sublayer)
            module.is_target_layer = False
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    def generate_image_for_specific_layer(target_layer, prompt, negative_prompt,seed, num_inference_steps, guidance_scale):
            """为特定层生成图像。"""
            logging.info(f"Generating image for layer: {target_layer}")

            for layer, sublayers in layers.items():
                for sublayer in sublayers:
                    module = pipe.unet.get_submodule(sublayer)
                    module.is_target_layer = (layer == target_layer)

            with torch.no_grad():
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]

            return image

    return generate_image_for_specific_layer, layers, hooks
def create_comparison_image(images, output_dir, layer_order):
    """
    创建一个包含所有生成图像的比较图像，按照层的生成顺序排列，合成图放在最后。
    
    :param images: 字典，键为层名称，值为对应的图像
    :param output_dir: 输出目录
    :param layer_order: 层的顺序列表
    """
    # 确保所有层都有对应的图像
    assert set(images.keys()) == set(layer_order) | {'all_layers'}, "图像和层顺序不匹配"
    
    # 按照层顺序排列图像，将合成图放在最后
    ordered_images = [images[layer] for layer in layer_order] + [images['all_layers']]
    
    # 计算总宽度和最大高度
    widths, heights = zip(*(i.size for i in ordered_images))
    total_width = sum(widths)
    max_height = max(heights)

    # 创建新的拼接图像
    comparison = Image.new('RGB', (total_width, max_height))
    
    # 拼接图像
    x_offset = 0
    for img in ordered_images:
        comparison.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # 创建比较图像的子目录
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_path = os.path.join(comparison_dir, "layer_comparison.png")
    comparison.save(comparison_path)
    logging.info(f"Saved comparison image to {comparison_path}")

    return comparison
def main():
    """主函数，运行整个图像生成和比较过程。"""
    parser = argparse.ArgumentParser(description='逐层生成图像并比较交叉注意力层的影响。')
    parser.add_argument('--seed', type=int, default=469, help='随机种子以确保可重现性')
    parser.add_argument('--prompt', type=str, default='A black cat nestled on a table, abstract style', help='用于图像生成的提示词')
    parser.add_argument('--negative_prompt', type=str, default='without multiple maple leaves', help='用于图像生成的负提示词')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步骤的数量')
    parser.add_argument('--guidance_scale', type=float, default=10.0, help='图像生成的指导尺度')
    parser.add_argument('--output_dir', type=str, default='layer_outputs', help='保存输出图像的目录')
    parser.add_argument('--model_path', type=str, default='../examples/DREAMBOOTH/SDXL_model', help='本地SDXL模型路径')
    parser.add_argument('--vae_path', type=str, default='../examples/DREAMBOOTH/VAE', help='本地vae模型路径')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32, local_files_only=True)
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float32)
        pipe.vae = vae
        pipe = pipe.to(device)
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return

    generate_image_for_specific_layer, layers, hooks = modify_pipeline_for_layer_specific_cross_attention(pipe)

    os.makedirs(args.output_dir, exist_ok=True)

    # 获取排序后的层
    sorted_layers = sort_layers(layers)
    layer_order = list(sorted_layers.keys())

    layer_images = {}
    for layer_name in layer_order:
        try:
            image = generate_image_for_specific_layer(layer_name, args.prompt, args.negative_prompt, args.seed, args.num_inference_steps, args.guidance_scale)
            # safe_layer_name = layer_name.replace('.', '_')
            safe_layer_name = layer_name
            image_path = os.path.join(args.output_dir, f"{safe_layer_name}.png")
            image.save(image_path)
            layer_images[layer_name] = image
            # logging.info(f"Generated image for layer {layer_name}")
        except Exception as e:
            logging.error(f"Error generating image for layer {layer_name}: {e}")

    try:
        for hook in hooks:
            hook.remove()

        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        all_layers_image = pipe(args.prompt,negative_prompt=args.negative_prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, generator=generator).images[0]
        all_layers_image.save(os.path.join(args.output_dir, "all_layers.png"))
        layer_images['all_layers'] = all_layers_image
        logging.info("Generated image using all layers")
    except Exception as e:
        logging.error(f"Error generating image for all layers: {e}")

    try:
        create_comparison_image(layer_images, args.output_dir, layer_order)
    except Exception as e:
        logging.error(f"Error creating comparison image: {e}")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
if __name__ == "__main__":
    main()
