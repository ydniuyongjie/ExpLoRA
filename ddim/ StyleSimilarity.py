#-----------------------0.9308----------------
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class StyleSimilarity:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 选择用于提取风格特征的层
        self.style_layers = ['0', '5', '10', '19', '28']
        
    def extract_features(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        features = []
        for name, layer in self.model._modules.items():
            image = layer(image)
            if name in self.style_layers:
                features.append(self._gram_matrix(image))
        return features
    
    def _gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    
    def calculate_similarity(self, image1, image2):
        features1 = self.extract_features(image1)
        features2 = self.extract_features(image2)
        
        similarity = 0
        for f1, f2 in zip(features1, features2):
            similarity += torch.nn.functional.cosine_similarity(f1.flatten(), f2.flatten(), dim=0)
        
        return similarity.item() / len(self.style_layers)

def main():
    # 使用示例
    style_similarity = StyleSimilarity()
    
    # 加载两张图片
    image1 = Image.open('a7.png')
    image2 = Image.open('all_layers.png')
    
    # 计算风格相似度
    similarity = style_similarity.calculate_similarity(image1, image2)
    print(f"风格相似度: {similarity:.4f}")

if __name__ == '__main__':
    main()



#-------------------------------------------------------
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from transformers import ViTFeatureExtractor, ViTModel
# import numpy as np

# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载预训练的ViT模型
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224')
# model.to(device)
# model.eval()

# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     inputs = feature_extractor(images=image, return_tensors="pt").to(device)
#     return inputs

# def get_vit_features(image_path):
#     inputs = load_and_preprocess_image(image_path)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # 使用最后一层的隐藏状态作为特征
#     features = outputs.last_hidden_state.squeeze(0)
#     return features

# def gram_matrix(tensor):
#     # 适应ViT输出的形状
#     if len(tensor.shape) == 2:
#         # 如果是2D张量，直接计算Gram矩阵
#         gram = torch.mm(tensor, tensor.t())
#     else:
#         # 如果是3D张量，首先将其reshape
#         b, n, c = tensor.shape
#         tensor = tensor.reshape(n, c)
#         gram = torch.mm(tensor.t(), tensor)
    
#     # 归一化
#     return gram / tensor.numel()

# def calculate_style_similarity(features1, features2):
#     # 计算Gram矩阵
#     gram1 = gram_matrix(features1)
#     gram2 = gram_matrix(features2)
    
#     # 使用余弦相似度计算风格相似性
#     similarity = torch.nn.functional.cosine_similarity(gram1.flatten(), gram2.flatten(), dim=0)
#     return similarity.item()

# def normalize_similarity(similarity):
#     # 将相似度归一化到0-100的范围
#     return (similarity + 1) * 50

# def main(image_path1, image_path2):
#     # 提取特征
#     features1 = get_vit_features(image_path1)
#     features2 = get_vit_features(image_path2)
    
#     # 计算风格相似性
#     similarity = calculate_style_similarity(features1, features2)
    
#     # 归一化相似度分数
#     normalized_similarity = normalize_similarity(similarity)
    
#     print(f"原始风格相似度分数: {similarity:.4f}")
#     print(f"归一化风格相似度分数 (0-100): {normalized_similarity:.2f}")

# if __name__ == "__main__":
#     image_path1 = "all_layers.png"
#     image_path2 = "a7.png"
#     main(image_path1, image_path2)
#---------------------------------0.9172------
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image

# # 加载和预处理图像
# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),  # 调整大小
#         transforms.ToTensor(),           # 转换为Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
#     ])
#     image = preprocess(image).unsqueeze(0)  # 增加批次维度
#     return image

# # 提取图像特征
# def extract_features(image_path, model):
#     image = load_and_preprocess_image(image_path)
#     with torch.no_grad():  # 禁用梯度计算
#         features = model(image)  # 特征形状为(N, C, H, W)
#     return features.view(features.size(0), -1)  # 展平为2D张量 (N, C*H*W)

# # 计算余弦相似度
# def cosine_similarity(features1, features2):
#     features1 = features1 / features1.norm(dim=1, keepdim=True)  # 归一化特征向量
#     features2 = features2 / features2.norm(dim=1, keepdim=True)
#     similarity = torch.mm(features1, features2.t())  # 计算余弦相似度
#     return ((similarity + 1) / 2).item()  # 线性变换到 0-1 范围


# # 主程序
# if __name__ == "__main__":
#     # 使用预训练的ResNet模型（去掉最后的全连接层）
#     model = models.resnet50(pretrained=True)
#     model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后一层
#     model.eval()  # 设置为评估模式

#     # 输入两张图片的路径
#     image1_path = 'all_layers.png'  # 替换为你的图片路径
#     image2_path = 'a7.png'  # 替换为你的图片路径

#     # 提取特征并计算相似度
#     features1 = extract_features(image1_path, model)
#     features2 = extract_features(image2_path, model)

#     similarity = cosine_similarity(features1, features2)
    
#     print(f'Cosine Similarity between the two images: {similarity:.4f}')

#----------------------------------0.6778-------------
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import lpips

# class StyleExtractor(nn.Module):
#     def __init__(self):
#         super(StyleExtractor, self).__init__()
#         vgg19 = models.vgg19(pretrained=True).features
#         self.features = nn.Sequential(*list(vgg19.children())[:18])
#         for param in self.features.parameters():
#             param.requires_grad_(False)
    
#     def forward(self, x):
#         return self.features(x)

# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     return transform(image).unsqueeze(0)

# def gram_matrix(input):
#     a, b, c, d = input.size()
#     features = input.view(a * b, c * d)
#     G = torch.mm(features, features.t())
#     return G.div(a * b * c * d)

# def extract_style_features(image, model):
#     features = model(image)
#     return gram_matrix(features)

# def style_similarity(gram1, gram2):
#     return nn.functional.cosine_similarity(gram1.flatten().unsqueeze(0), gram2.flatten().unsqueeze(0)).item()

# def compare_style(image1_path, image2_path):
#     # VGG19 for Gram Matrix
#     style_extractor = StyleExtractor()
#     style_extractor.eval()

#     # LPIPS
#     lpips_model = lpips.LPIPS(net='vgg')

#     # Load and preprocess images
#     image1 = preprocess_image(image1_path)
#     image2 = preprocess_image(image2_path)

#     with torch.no_grad():
#         # Gram Matrix similarity
#         style_features1 = extract_style_features(image1, style_extractor)
#         style_features2 = extract_style_features(image2, style_extractor)
#         gram_similarity = style_similarity(style_features1, style_features2)

#         # LPIPS similarity
#         lpips_distance = lpips_model(image1, image2).item()
#         lpips_similarity = 1 - lpips_distance

#     # Combine similarities
#     combined_similarity = (gram_similarity + lpips_similarity) / 2

#     return combined_similarity

# if __name__ == "__main__":
#     image1_path = 'a7.png'
#     image2_path = 'all_layers.png'

#     similarity = compare_style(image1_path, image2_path)
#     print(f'Style similarity score: {similarity:.4f}')
