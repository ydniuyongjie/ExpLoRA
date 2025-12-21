

# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torchvision.models import VGG19_Weights
# from PIL import Image
# from scipy.spatial.distance import cosine

# # 加载预训练的VGG19模型
# style_model = models.vgg19(weights=VGG19_Weights.DEFAULT).features
# style_model.eval()
# #VGG19_Weights.IMAGENET1K_V1
# # 图像预处理
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def compute_style_features(image, model):
#     # 提取风格特征
#     img_tensor = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         features = []
#         for layer in model:
#             img_tensor = layer(img_tensor)
#             if isinstance(layer, torch.nn.Conv2d):
#                 features.append(img_tensor)
#     gram_matrices = [gram_matrix(feat) for feat in features]
#     return torch.cat([g.view(-1) for g in gram_matrices])

# def gram_matrix(input):
#     # 计算Gram矩阵
#     a, b, c, d = input.size()
#     features = input.view(a * b, c * d)
#     G = torch.mm(features, features.t())
#     return G.div(a * b * c * d)

# def compare_styles(style_a, style_b):
#     # 比较风格特征的相似度
#     return 1 - cosine(style_a.numpy(), style_b.numpy())

# def analyze_style_similarity(image_a_path, image_b_path):
#     image_a = Image.open(image_a_path).convert('RGB')
#     image_b = Image.open(image_b_path).convert('RGB')

#     # 提取风格特征
#     style_a = compute_style_features(image_a, style_model)
#     style_b = compute_style_features(image_b, style_model)

#     # 比较风格相似度
#     style_similarity = compare_styles(style_a, style_b)

#     return style_similarity

# def main():
#     # 替换为你的实际图片路径
#     image_a_path = 'base.png'
#     image_b_path = 'B.png'

#     try:
#         style_sim = analyze_style_similarity(image_a_path, image_b_path)
#         print(f"Style similarity: {style_sim:.2f}")
    
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()

#-----------------------------------------------------------------------------------------------#
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import Resize
from scipy.spatial.distance import cosine
import clip

# 加载 SAM 模型
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-L/14", device=device)

def get_image_from_path(path):
    return Image.open(path).convert("RGB")

def extract_contour(image):
    predictor.set_image(np.array(image))
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=False,
    )
    contour = masks[0].astype(np.uint8) * 255
    return Image.fromarray(contour)

def get_clip_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(image)
    return features.cpu().numpy()

def compare_contours(image1, image2):
    # 提取轮廓
    contour1 = extract_contour(image1)
    contour2 = extract_contour(image2)
    
    # 调整大小以匹配 CLIP 输入要求
    contour1 = Resize((224, 224))(contour1)
    contour2 = Resize((224, 224))(contour2)
    
    # 获取 CLIP 特征
    features1 = get_clip_features(contour1)
    features2 = get_clip_features(contour2)
    
    # 计算余弦相似度
    similarity = 1 - cosine(features1.flatten(), features2.flatten())
    return similarity

# 示例使用
image_path1 = "C.png"
image_path2 = "B.png"

try:
    image1 = get_image_from_path(image_path1)
    image2 = get_image_from_path(image_path2)

    similarity = compare_contours(image1, image2)
    print(f"Contour similarity: {similarity:.4f}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
#---------------------------------------------------------




#只使用了 layer2, layer3, 和 layer4 的输出，这些层捕获了中高级特征，可能更适合判断整体内容相似性。
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.conv1 = nn.Sequential(*list(resnet.children())[:4])
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 计算正确的输入维度
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, 224, 224)
#             x1 = self.conv1(dummy_input)
#             x2 = self.layer1(x1)
#             x3 = self.layer2(x2)
#             x4 = self.layer3(x3)
#             x5 = self.layer4(x4)
#             features = [x2, x3, x4, x5]
#             features = [self.avgpool(f).view(f.size(0), -1) for f in features]
#             total_dim = sum(f.size(1) for f in features)
        
#         self.fc = nn.Linear(total_dim, 512)
#         self.bn = nn.BatchNorm1d(512)
        
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.layer1(x1)
#         x3 = self.layer2(x2)
#         x4 = self.layer3(x3)
#         x5 = self.layer4(x4)
        
#         features = [x2, x3, x4, x5]
#         features = [self.avgpool(f).view(f.size(0), -1) for f in features]
#         x = torch.cat(features, dim=1)
#         x = self.fc(x)
#         x = self.bn(x)
#         return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FeatureExtractor().to(device)
# model.eval()

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def extract_features(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = preprocess(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         features = model(image_tensor)
    
#     return features.squeeze().cpu().numpy()

# def calculate_similarity(features1, features2):
#     # 使用L2距离
#     distance = np.linalg.norm(features1 - features2)
    
#     # 将距离转换为相似度，并增强对比度
#     similarity = np.exp(-distance / 10)  # 可以调整这个因子
    
#     # 应用阈值
#     threshold = 0.5  # 可以根据需要调整
#     similarity = max(0, (similarity - threshold) / (1 - threshold))
    
#     return similarity

# def compare_images(image_path1, image_path2):
#     features1 = extract_features(image_path1)
#     features2 = extract_features(image_path2)
    
#     similarity = calculate_similarity(features1, features2)
#     return similarity

# # 使用示例
# image_path1 = "B.png"
# image_path2 = "A.png"

# try:
#     similarity = compare_images(image_path1, image_path2)
#     print(f"The content similarity between the two images is: {similarity:.4f}")
# except Exception as e:
#     print(f"An error occurred: {str(e)}")
