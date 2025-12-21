import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import cv2
import os
import pandas as pd

# class ObjectSimilarityEvaluator:
#     def __init__(self):
#         self.lpips_fn = lpips.LPIPS(net='alex').cuda()
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#     def preprocess_image(self, img):
#         if isinstance(img, np.ndarray):
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = self.transform(img).unsqueeze(0).cuda()
#         return img
        
#     def evaluate_shape_accuracy(self, img1, img2):
#         img1, img2 = self.preprocess_image(img1), self.preprocess_image(img2)
#         lpips_score = self.lpips_fn(img1, img2).item()
#         shape_score = (1 - lpips_score) * 10
#         return min(max(shape_score, 0), 10)
    
#     def evaluate_structural_completeness(self, img1, img2):
#         """评估结构完整性"""
#         try:
#             min_side = min(img1.shape[0], img1.shape[1])
#             win_size = 3  # 使用最小的窗口大小
#             ssim_score = ssim(img1, img2, win_size=win_size, channel_axis=2)
#             structure_score = ssim_score * 10
#             return min(max(structure_score, 0), 10)
#         except Exception as e:
#             print(f"SSIM计算错误: {str(e)}")
#             return 0  # 如果计算失败返回0分
    
#     def evaluate_detail_retention(self, img1, img2):
#         try:
#             edges1 = cv2.Canny(img1, 100, 200)
#             edges2 = cv2.Canny(img2, 100, 200)
#             win_size = 3  # 使用最小的窗口大小
#             edge_similarity = ssim(edges1, edges2, win_size=win_size, multichannel=False)
#             detail_score = edge_similarity * 10
#             return min(max(detail_score, 0), 10)
#         except Exception as e:
#             print(f"边缘相似度计算错误: {str(e)}")
#             return 0
    
#     def evaluate(self, layer_img, final_img):
#         shape_score = self.evaluate_shape_accuracy(layer_img, final_img)
#         structure_score = self.evaluate_structural_completeness(layer_img, final_img)
#         detail_score = self.evaluate_detail_retention(layer_img, final_img)
        
#         final_score = (
#             shape_score * 0.4 +
#             structure_score * 0.3 +
#             detail_score * 0.3
#         )
        
#         return {
#             'shape_accuracy': round(shape_score, 3),
#             'structural_completeness': round(structure_score, 3),
#             'detail_retention': round(detail_score, 3),
#             'final_score': round(final_score, 3)
#         }




class ObjectSimilarityEvaluator:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()  # 改用VGG backbone
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                               std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, img):
        if isinstance(img, np.ndarray):
            # 添加图像预处理步骤
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 确保图像大小适中
            if max(img.shape[:2]) > 512:
                scale = 512 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=scale, fy=scale)
            img = self.transform(img).unsqueeze(0).cuda()
        return img
        
    def evaluate_shape_accuracy(self, img1, img2):
        """改进的形状准确度评估"""
        # LPIPS评分
        img1, img2 = self.preprocess_image(img1), self.preprocess_image(img2)
        lpips_score = self.lpips_fn(img1, img2).item()
        
        # 添加MSE评分
        mse = np.mean((img1.cpu().numpy() - img2.cpu().numpy()) ** 2)
        mse_score = np.exp(-mse)
        
        # 综合评分
        shape_score = (0.7 * (1 - lpips_score) + 0.3 * mse_score) * 10
        return min(max(shape_score, 0), 10)
    
    def evaluate_structural_completeness(self, img1, img2):
        """改进的结构完整性评估"""
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 多尺度SSIM
            scores = []
            for win_size in [7, 11, 15]:  # 使用多个窗口大小
                score = ssim(gray1, gray2, 
                           win_size=win_size, 
                           gaussian_weights=True,
                           sigma=1.5,
                           use_sample_covariance=False)
                scores.append(score)
            
            # 取平均值
            structure_score = np.mean(scores) * 10
            return min(max(structure_score, 0), 10)
        except Exception as e:
            print(f"SSIM计算错误: {str(e)}")
            return 0
    
    def evaluate_detail_retention(self, img1, img2):
        """改进的细节保留度评估"""
        try:
            # 多尺度边缘检测
            def get_edges(img):
                edges = []
                for thresh in [(50,100), (100,200), (150,300)]:
                    edge = cv2.Canny(img, thresh[0], thresh[1])
                    edges.append(edge)
                return np.maximum.reduce(edges)
            
            edges1 = get_edges(img1)
            edges2 = get_edges(img2)
            
            # 计算边缘图的相似度
            edge_similarity = ssim(edges1, edges2, win_size=7, multichannel=False)
            
            # 添加梯度相似度
            def get_gradients(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                return np.sqrt(gx**2 + gy**2)
            
            grad1 = get_gradients(img1)
            grad2 = get_gradients(img2)
            grad_similarity = ssim(grad1, grad2, win_size=7, multichannel=False)
            
            # 综合评分
            detail_score = (0.6 * edge_similarity + 0.4 * grad_similarity) * 10
            return min(max(detail_score, 0), 10)
        except Exception as e:
            print(f"细节评估错误: {str(e)}")
            return 0
    
    def evaluate(self, layer_img, final_img):
        """评估总分"""
        shape_score = self.evaluate_shape_accuracy(layer_img, final_img)
        structure_score = self.evaluate_structural_completeness(layer_img, final_img)
        detail_score = self.evaluate_detail_retention(layer_img, final_img)
        
        # 调整权重
        final_score = (
            shape_score * 0.5 +      # 增加形状准确度的权重
            structure_score * 0.3 +
            detail_score * 0.2       # 降低细节评分的权重
        )
        
        return {
            'shape_accuracy': round(shape_score, 3),
            'structural_completeness': round(structure_score, 3),
            'detail_retention': round(detail_score, 3),
            'final_score': round(final_score, 3)
        }

def evaluate_all_layers(image_folder, final_image_name='all_layers.png'):
    """
    评估文件夹中所有层图像与最终图像的相似度
    
    参数:
        image_folder: 包含所有层图像的文件夹路径
        final_image_name: 最终图像的文件名
    """
    evaluator = ObjectSimilarityEvaluator()
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
                
            # 确保图像尺寸一致
            if layer_image.shape != final_image.shape:
                layer_image = cv2.resize(layer_image, (final_image.shape[1], final_image.shape[0]))
            
            # 评估相似度
            try:
                scores = evaluator.evaluate(layer_image, final_image)
                
                # 添加层名称
                results.append({
                    'layer_name': filename,
                    'shape_accuracy': scores['shape_accuracy'],
                    'structural_completeness': scores['structural_completeness'],
                    'detail_retention': scores['detail_retention'],
                    'final_score': scores['final_score']
                })
            except Exception as e:
                print(f"评估图像 {filename} 时出错: {str(e)}")
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(results)
    df = df.sort_values('final_score', ascending=False)
    
    return df

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
        
        # 获取按层排序后相似度最高的前3层
        top_3_layers = sorted_results_df.nlargest(3, 'final_score')
        
        print("\n相似度最高的前3层:")
        for _, row in top_3_layers.iterrows():
            layer_number = layer_mapping[row['layer_name']]
            print(f"层序号: {layer_number}")
            print(f"文件名: {row['layer_name']}")
            print(f"最终得分: {row['final_score']:.3f}")
            print(f"形状准确度: {row['shape_accuracy']:.3f}")
            print(f"结构完整性: {row['structural_completeness']:.3f}")
            print(f"细节保留度: {row['detail_retention']:.3f}")
            print("-" * 50)
            
        # 创建用于显示的DataFrame副本
        display_df = results_df.copy()
        # 替换文件名中的点为下划线
        display_df['layer_name'] = display_df['layer_name'].str.replace('.', '_')
        # 按final_score降序排序并重置索引
        display_df = display_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # 打印完整的评估结果表格
        print("\nObject Similarity Evaluation Results:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print(display_df[['layer_name', 'shape_accuracy', 'structural_completeness', 
                         'detail_retention', 'final_score']])
        
        # 保存原始结果到CSV（使用点作为分隔符的版本）
        sorted_results_df = sorted_results_df.drop('sort_key', axis=1)
        sorted_results_df.to_csv('layer_similarity_scores.csv', index=False)
        
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
