import argparse
import json
import os
import shutil

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from accelerate.utils import  set_seed
from diffusers import AutoencoderKL, StableDiffusionXLPipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a inference  script.")
    parser.add_argument(
        "--lora_model_id",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained lora or lora identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--new_prompt",
        type=str,
        default=None,
        help=("The prompt or prompts to guide the image generation."),
    )
    parser.add_argument(
        "--old_prompt",
        type=str,
        default=None,
        help=("The prompt or prompts to guide the image generation."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=("The directory where to save the exported images."),
    )
    parser.add_argument(
        "--num_images",
        default=4,
        help="The number of images",
        type=int,
    )
    args = parser.parse_args(input_args)
    return args
def load_combined_lora_weights(pipe, weights_config):
    """
    weights_config = [
        {
            'path': 'path_A',
            'layers': ['down_blocks.1', 'down_blocks.2']
        },
        {
            'path': 'path_B',
            'layers': ['down_blocks.0']
        }
    ]
    """
    combined_weights = {}
    with open(f"{weights_config[0]['path']}/lora_config.json", "r") as f:
            config_dict = json.load(f)
    # 转换 target_modules
    target_modules = convert_to_list(config_dict.get("target_modules"))
    # 创建 LoraConfig，确保包含 RSLoRA 参数
    lora_config = LoraConfig(
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config_dict["lora_dropout"],
        bias=config_dict["bias"],
        use_rslora=config_dict["use_rslora"],
        use_active_func=config_dict["use_active_func"],
    )
    # 应用 LoRA 配置到 U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    # 遍历每个权重配置
    for config in weights_config:
        # 加载权重文件
        weights = load_file(f"{config['path']}/unet_lora_weights.safetensors")
        weights = remap_lora_keys(weights)
        
        # 提取指定层的权重
        for key, value in weights.items():
            if any(layer in key for layer in config['layers']):
                combined_weights[key] = value
    
    # 加载组合后的权重
    pipe.unet.load_state_dict(combined_weights, strict=False)
    
    return pipe

# ---------------------------使用示例----------------------
# weights_config = [
#     {
#         'path': 'path_A',
#         'layers': ['down_blocks.1', 'down_blocks.2']
#     },
#     {
#         'path': 'path_B',
#         'layers': ['down_blocks.0']
#     }
# ]

# pipe = load_combined_lora_weights(pipe, weights_config)
#----------------------------------------------------------------------------------

def convert_to_list(s):
    # 移除花括号，分割字符串，并去除引号
    return [item.strip("'") for item in s.strip("{}").split(", ")]

def adjust_model_scales(model, mode="object"):
        """调整模型中所有LoRA层的scaling
        Args:
            model: 带有LoRA的模型
            mode: 'object' 或 'style'
        """
        OBJECT_SCALE_4 = 4.4 #3.4  # 使用您计算好的值
        OBJECT_SCALE_6 =4.8 #3.8  # 使用您计算好的值
        STYLE_SCALE = 4.0  # 使用您计算好的值

        for name, module in model.named_modules():
            if hasattr(module, "set_scale"):
                adapter_name = module._active_adapter[0]  # 转换为字符串
                if mode == "object":
                    if "down_blocks.2.attentions.1" in name:  # 对象特征层
                        module.set_scale(adapter_name, OBJECT_SCALE_4)
                    elif "up_blocks.0.attentions.0" in name:
                        module.set_scale(adapter_name, OBJECT_SCALE_6)
                    else:
                        module.set_scale(adapter_name, 1.0)
                if mode == "style": # style mode
                    if "up_blocks.0.attentions.1" in name:  # 风格特征层
                        module.set_scale(adapter_name, STYLE_SCALE)
                    else:
                        module.set_scale(adapter_name, 1.0)
                if mode=="both":
                    if "down_blocks.2.attentions.1" in name:
                        module.set_scale(adapter_name, OBJECT_SCALE_4)
                    elif "up_blocks.0.attentions.0" in name:
                        module.set_scale(adapter_name, OBJECT_SCALE_6)
                    elif "up_blocks.0.attentions.1" in name:
                        module.set_scale(adapter_name, STYLE_SCALE)
                    else:
                        module.set_scale(adapter_name, 0.0)
    
def check_lora_scales(model):
    """获取LoRA层的缩放系数
    Args:
    model: 带有LoRA的模型
    Returns:
    None
    """
    scales = {}
    key_mapping = {
        "down_blocks.2.attentions.1": "4层",
        "up_blocks.0.attentions.0": "6层", 
        "up_blocks.0.attentions.1": "7层"
    }
    
    for name, module in model.named_modules():
        if hasattr(module, "get_scale"):
            for key in key_mapping:
                if key in name:
                    scales[key_mapping[key]] = module.get_scale(module._active_adapter[0])
    
    print("放大系数为:", scales)
    # 比如输出: 放大系数为: {'4': 0.5, '6': 0.7, '7': 0.6}        
    # return scales

def remap_lora_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'lora' in key:
            # 添加前缀并在 weight 前插入 default
            new_key = f"base_model.model.{key.rsplit('.weight', 1)[0]}.default.weight"
            new_state_dict[new_key] = value
        else:
            # 对于非 LoRA 权重，保持原样
            new_state_dict[key] = value
    return new_state_dict
def load_lora_model(pipe, load_path):
    # 加载 LoRA 配置
    with open(f"{load_path}/lora_config.json", "r") as f:
        config_dict = json.load(f)
    # 转换 target_modules
    target_modules = convert_to_list(config_dict.get("target_modules"))
    # 创建 LoraConfig，确保包含 RSLoRA 参数
    lora_config = LoraConfig(
        r=config_dict["r"],
        lora_alpha=config_dict["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=config_dict["lora_dropout"],
        bias=config_dict["bias"],
        use_rslora=config_dict["use_rslora"],
        use_active_func=config_dict["use_active_func"],
    )
    # 应用 LoRA 配置到 U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    # 加载 U-Net LoRA 权重
    unet_lora_layers = load_file(f"{load_path}/unet_lora_weights.safetensors")

    # for key in pipe.unet.state_dict().keys():
    #     print(key)
    # # print("Model keys:", pipe.unet.state_dict().keys())
    # # print("Loaded keys:", unet_lora_layers.keys())
    # for key in unet_lora_layers.keys():
    #     print(key,unet_lora_layers[key])
    new_unet_lora_layers = remap_lora_keys(unet_lora_layers)
    # for key in new_unet_lora_layers.keys():
    #     print(key,new_unet_lora_layers[key])
    pipe.unet.load_state_dict(new_unet_lora_layers, strict=False)
    print(f"LoRA model loaded from {load_path}")

    return pipe

if __name__ == "__main__":
    args = parse_args()
    set_seed(0)
    lora_model_id = args.lora_model_id
    base_model = "SDXL_model"
    VAE_model = "VAE"

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(VAE_model, torch_dtype=torch.float32)
    pipe.vae = vae
    #加载模型的配置文件和权重
    # pipe = load_lora_model(pipe, lora_model_id)
    weights_config = [
        {
            'path': 'lora_obj',
            'layers': ['down_blocks.2.attentions.1', 'up_blocks.0.attentions.0']
        },
        {
            'path': 'lora_sty',
            'layers': ['up_blocks.0.attentions.1']
        }
    ]

    pipe = load_combined_lora_weights(pipe, weights_config)
    pipe.to("cuda")
    adjust_model_scales(pipe.unet, mode="both")
    check_lora_scales(pipe.unet)

    # new prompt
    prompt = args.new_prompt
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir+'/new/')
    images = [pipe(prompt, num_inference_steps=25).images[0] for i in range(args.num_images)]
    for i, image in enumerate(images):
        image.save(f"{output_dir+'/new/'}/dog{i}-343840_900_400.png")
    # old prompt
    print("old prompt-------------------")
    prompt = args.old_prompt
    output_dir = args.output_dir
    os.makedirs(output_dir+'/old/')
    images = [pipe(prompt, num_inference_steps=25).images[0] for i in range(1)]
    for i, image in enumerate(images):
        image.save(f"{output_dir+'/old/'}/cat{i}.png")

