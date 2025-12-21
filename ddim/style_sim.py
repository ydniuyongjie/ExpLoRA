import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class StyleSimilarityEvaluator:
    def __init__(self):
        # 初始化VGG19模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = models.vgg19(pretrained=True).features.eval().to(self.device)
        
        # 定义特征层
        self.style_layers = {
            'low': ['0', '5'],          # 低级特征
            'mid': ['10', '19'],        # 中级特征
            'high': ['28', '34']        # 高级特征
        }
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 评分权重
        self.weights = {
            'low_features': 0.15,
            'mid_features': 0.20,
            'high_features': 0.15,
            'color': 0.15,
            'texture': 0.15,
            'composition': 0.10
        }

    def preprocess_image(self, img):
        """图像预处理"""
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))  # 确保尺寸一致
            img = self.transform(img).unsqueeze(0).to(self.device)
        return img

    def extract_features(self, x):
        """提取VGG特征"""
        features = {'low': [], 'mid': [], 'high': []}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            for level, layers in self.style_layers.items():
                if name in layers:
                    features[level].append(x.detach())
        return features

    def gram_matrix(self, x):
        """计算Gram矩阵"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def color_analysis(self, img1, img2):
        """颜色分析"""
        def get_color_features(img):
            # 确保图像大小一致
            img = cv2.resize(img, (224, 224))
            
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 计算每个通道的直方图
            features = []
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.extend(hist)
            
            return np.array(features, dtype=np.float32)

        try:
            f1 = get_color_features(img1)
            f2 = get_color_features(img2)
            
            # 计算余弦相似度
            similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]
            color_score = max(0, min(10, similarity * 10))
            
            return float(color_score)
        except Exception as e:
            print(f"颜色分析错误: {str(e)}")
            return 5.0

    def texture_analysis(self, img1, img2):
        """纹理分析"""
        def get_texture_features(img):
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            # 计算LBP特征
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2), density=True)
            
            return hist

        try:
            f1 = get_texture_features(img1)
            f2 = get_texture_features(img2)
            
            # 计算余弦相似度
            similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]
            texture_score = max(0, min(10, similarity * 10))
            
            return float(texture_score)
        except Exception as e:
            print(f"纹理分析错误: {str(e)}")
            return 5.0

    def composition_analysis(self, img1, img2):
        """构图分析"""
        def get_composition_features(img):
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            # 计算边缘
            edges = cv2.Canny(gray, 100, 200)
            edge_hist = cv2.calcHist([edges], [0], None, [8], [0, 256])
            edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
            
            # 计算图像的分块统计特征
            features = []
            h, w = gray.shape
            block_size = h // 4
            for i in range(4):
                for j in range(4):
                    block = gray[i*block_size:(i+1)*block_size, 
                               j*block_size:(j+1)*block_size]
                    features.extend([np.mean(block), np.std(block)])
            
            return np.concatenate([edge_hist, features])

        try:
            f1 = get_composition_features(img1)
            f2 = get_composition_features(img2)
            
            # 计算余弦相似度
            similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]
            composition_score = max(0, min(10, similarity * 10))
            
            return float(composition_score)
        except Exception as e:
            print(f"构图分析错误: {str(e)}")
            return 5.0

    def evaluate(self, layer_img, final_img):
        """评估单个图像的样式相似度"""
        try:
            # 提取VGG特征
            layer_tensor = self.preprocess_image(layer_img)
            final_tensor = self.preprocess_image(final_img)
            
            layer_features = self.extract_features(layer_tensor)
            final_features = self.extract_features(final_tensor)
            
            # 计算各层特征的相似度
            feature_scores = {}
            for level in ['low', 'mid', 'high']:
                score = 0
                for f1, f2 in zip(layer_features[level], final_features[level]):
                    gram1 = self.gram_matrix(f1)
                    gram2 = self.gram_matrix(f2)
                    diff = torch.mean(torch.abs(gram1 - gram2)).item()
                    score += diff
                feature_scores[level] = max(0, min(10, 10 * np.exp(-5 * score)))
            
            # 计算其他特征得分
            color_score = self.color_analysis(layer_img, final_img)
            texture_score = self.texture_analysis(layer_img, final_img)
            composition_score = self.composition_analysis(layer_img, final_img)
            
            # 计算加权总分
            style_score = (
                self.weights['low_features'] * feature_scores['low'] +
                self.weights['mid_features'] * feature_scores['mid'] +
                self.weights['high_features'] * feature_scores['high'] +
                self.weights['color'] * color_score +
                self.weights['texture'] * texture_score +
                self.weights['composition'] * composition_score
            )
            
            return {
                'detail_scores': {
                    'vgg_features': feature_scores,
                    'color': color_score,
                    'texture': texture_score,
                    'composition': composition_score
                },
                'style_score': style_score
            }
            
        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            return None

def evaluate_all_layers(image_folder, final_image_name='all_layers.png'):
    """评估所有层的样式得分"""
    evaluator = StyleSimilarityEvaluator()
    results = []
    
    # 读取最终图像
    final_image_path = os.path.join(image_folder, final_image_name)
    final_image = cv2.imread(final_image_path)
    
    if final_image is None:
        raise ValueError(f"无法读取最终图像: {final_image_path}")
    
    # 遍历文件夹中的所有图像
    for filename in os.listdir(image_folder):
        if filename == final_image_name:
            continue
            
        if filename.endswith('.png'):
            layer_path = os.path.join(image_folder, filename)
            layer_image = cv2.imread(layer_path)
            
            if layer_image is None:
                print(f"无法读取图像: {layer_path}")
                continue
            
            print(f"正在处理: {filename}")
            scores = evaluator.evaluate(layer_image, final_image)
            
            if scores is not None:
                results.append({
                    'layer_name': filename,
                    'vgg_low': scores['detail_scores']['vgg_features']['low'],
                    'vgg_mid': scores['detail_scores']['vgg_features']['mid'],
                    'vgg_high': scores['detail_scores']['vgg_features']['high'],
                    'color': scores['detail_scores']['color'],
                    'texture': scores['detail_scores']['texture'],
                    'composition': scores['detail_scores']['composition'],
                    'style_score': scores['style_score']
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    image_folder = "./layer_outputs_469_50_10_red_cube/layout_image"
    
    try:
        # 评估所有层
        results_df = evaluate_all_layers(image_folder)
        
        # 定义排序函数
        def get_layer_order(layer_name):
            # 移除.png后缀并分割名称
            name = layer_name.replace('.png', '')
            parts = name.split('.')
            
            try:
                if parts[0] == 'mid_block':
                    return (1, 0, 0)
                elif parts[0] == 'down_blocks':
                    return (0, int(parts[1]), int(parts[3]))
                elif parts[0] == 'up_blocks':
                    return (2, int(parts[1]), int(parts[3]))
                else:
                    return (3, 0, 0)  # 为未知类型提供默认排序
            except Exception as e:
                print(f"解析层名称出错 {layer_name}: {str(e)}")
                return (3, 0, 0)
        
        # 添加排序键并排序
        results_df['sort_key'] = results_df['layer_name'].apply(get_layer_order)
        sorted_results_df = results_df.sort_values('sort_key')
        
        # 添加层序号
        layer_count = 1
        layer_mapping = {}
        for idx, row in sorted_results_df.iterrows():
            layer_mapping[row['layer_name']] = f"第{layer_count}层"
            layer_count += 1
        
        # 获取按层排序后样式得分最高的前3层
        top_3_layers = sorted_results_df.nlargest(3, 'style_score')
        
        print("\n样式得分最高的前3层:")
        for _, row in top_3_layers.iterrows():
            layer_number = layer_mapping[row['layer_name']]
            print(f"层序号: {layer_number}")
            print(f"文件名: {row['layer_name']}")
            print(f"样式得分: {row['style_score']:.3f}")
            print(f"VGG低级特征: {row['vgg_low']:.3f}")
            print(f"VGG中级特征: {row['vgg_mid']:.3f}")
            print(f"VGG高级特征: {row['vgg_high']:.3f}")
            print(f"颜色得分: {row['color']:.3f}")
            print(f"纹理得分: {row['texture']:.3f}")
            print(f"构图得分: {row['composition']:.3f}")
            print("-" * 50)
            
        # 创建用于显示的DataFrame副本
        display_df = results_df.copy()
        # 替换文件名中的点为下划线
        display_df['layer_name'] = display_df['layer_name'].str.replace('.', '_')
        # 按style_score降序排序并重置索引
        display_df = display_df.sort_values('style_score', ascending=False).reset_index(drop=True)
        
        # 打印完整的评估结果表格
        print("\nStyle Similarity Evaluation Results:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print(display_df[['layer_name', 'vgg_low', 'vgg_mid', 'vgg_high', 
                         'color', 'texture', 'composition', 'style_score']])
        
        # 保存原始结果到CSV（使用点作为分隔符的版本）
        sorted_results_df = sorted_results_df.drop('sort_key', axis=1)
        sorted_results_df.to_csv('layer_style_scores.csv', index=False)
        
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
