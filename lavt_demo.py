#!/usr/bin/env python3
"""
LAVT: Language-Aware Vision Transformer for Referring Image Segmentation

使用方法:
python lavt_demo.py --input_image_path /path/to/image --prompt "Object description" --output_image_path /path/to/visualization
"""

import argparse
import os
import sys
from typing import Dict, Any
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation

# 导入项目相关模块
from bert.tokenization_bert import BertTokenizer
from bert.modeling_bert import BertModel
from lib import segmentation


class LAVTDemo:
    """LAVT演示类，用于加载模型和处理图像"""
    
    def __init__(self, model_path: str, device: str = 'cuda:0', model_type: str = 'lavt'):
        """
        初始化LAVT演示
        
        Args:
            model_path: 预训练模型权重路径
            device: 设备类型 ('cuda:0', 'cpu')
            model_type: 模型类型 ('lavt', 'lavt_one')
        """
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        
        # 检查设备可用性
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            self.device = 'cpu'
        
        # 初始化tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        except Exception as e:
            print(f"警告: 无法加载本地BERT模型: {e}")
            self.tokenizer = BertTokenizer.from_pretrained('./bert')
        
        # 初始化模型
        self._load_model()
        
        # 图像预处理
        self.image_transforms = T.Compose([
            T.Resize(480),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """加载预训练模型"""
        print(f"正在加载模型: {self.model_path}")
        
        # 创建参数配置
        class Args:
            swin_type = 'base'
            window12 = True
            mha = ''
            fusion_drop = 0.0
        
        args = Args()
        
        # 初始化模型
        if self.model_type == 'lavt':
            self.model = segmentation.__dict__['lavt'](pretrained='', args=args)
            try:
                self.bert_model = BertModel.from_pretrained('./bert-base-uncased')
            except Exception as e:
                print(f"警告: 无法加载本地BERT模型: {e}")
                self.bert_model = BertModel.from_pretrained('./bert')
            self.bert_model.pooler = None
        elif self.model_type == 'lavt_one':
            self.model = segmentation.__dict__['lavt_one'](pretrained='', args=args)
            self.bert_model = None
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 加载预训练权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        if self.model_type == 'lavt':
            self.bert_model.load_state_dict(checkpoint['bert_model'])
            self.model.load_state_dict(checkpoint['model'])
            self.bert_model = self.bert_model.to(self.device)
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.bert_model:
            self.bert_model.eval()
        
        print("模型加载完成")
    
    def _preprocess_text(self, description: str) -> tuple:
        """预处理文本描述"""
        sentence_tokenized = self.tokenizer.encode(text=description, add_special_tokens=True)
        sentence_tokenized = sentence_tokenized[:20]
        
        padded_sent_toks = [0] * 20
        padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
        
        attention_mask = [0] * 20
        attention_mask[:len(sentence_tokenized)] = [1] * len(sentence_tokenized)
        
        padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)
        
        return padded_sent_toks, attention_mask
    
    def _preprocess_image(self, image: Image.Image) -> tuple:
        """预处理图像"""
        original_w, original_h = image.size
        original_array = np.array(image)
        processed_image = self.image_transforms(image).unsqueeze(0).to(self.device)
        
        return processed_image, (original_h, original_w), original_array
    
    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray, 
                     colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4) -> np.ndarray:
        """将分割掩码叠加到图像上"""
        colors = np.reshape(colors, (-1, 3))
        colors = np.atleast_2d(colors) * cscale
        
        im_overlay = image.copy()
        object_ids = np.unique(mask)
        
        for object_id in object_ids[1:]:
            foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[object_id])
            binary_mask = mask == object_id
            
            im_overlay[binary_mask] = foreground[binary_mask]
            
            contours = binary_dilation(binary_mask) ^ binary_mask
            im_overlay[contours, :] = 0
        
        return im_overlay.astype(image.dtype)
    
    def forward(self, model: Any, image: Image.Image, description: str) -> Dict:
        """核心推理函数"""
        with torch.no_grad():
            # 预处理图像
            processed_image, original_size, original_array = self._preprocess_image(image)
            
            # 预处理文本
            padded_sent_toks, attention_mask = self._preprocess_text(description)
            
            # 模型推理
            if self.model_type == 'lavt':
                last_hidden_states = self.bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = self.model(processed_image, embedding, l_mask=attention_mask.unsqueeze(-1))
            else:
                output = self.model(processed_image, padded_sent_toks, attention_mask=attention_mask)
            
            # 后处理
            output = output.argmax(1, keepdim=True)
            output = F.interpolate(output.float(), original_size, mode='nearest')
            output = output.squeeze()
            output = output.cpu().data.numpy()
            output = output.astype(np.uint8)
            
            # 创建可视化结果
            visualization = self._overlay_mask(original_array, output)
            
            return {
                'mask': output,
                'visualization': Image.fromarray(visualization),
                'original_size': original_size,
                'original_array': original_array
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LAVT Referring Image Segmentation Demo')
    parser.add_argument('--input_image_path', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--prompt', type=str, required=True,
                        help='对象描述文本')
    parser.add_argument('--output_image_path', type=str, required=True,
                        help='输出可视化图像路径')
    parser.add_argument('--model_path', type=str, default='./checkpoints/refcoco.pth',
                        help='预训练模型权重路径')
    parser.add_argument('--model_type', type=str, default='lavt', choices=['lavt', 'lavt_one'],
                        help='模型类型')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备类型 (cuda:0, cpu)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_image_path):
        print(f"错误: 输入图像文件不存在: {args.input_image_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型权重文件不存在: {args.model_path}")
        print("请下载预训练权重到checkpoints目录")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 初始化演示类
        demo = LAVTDemo(
            model_path=args.model_path,
            device=args.device,
            model_type=args.model_type
        )
        
        # 加载图像
        print(f"正在处理图像: {args.input_image_path}")
        image = Image.open(args.input_image_path).convert("RGB")
        
        # 执行推理
        print(f"正在处理描述: '{args.prompt}'")
        result = demo.forward(None, image, args.prompt)
        
        # 保存结果
        result['visualization'].save(args.output_image_path)
        print(f"结果已保存到: {args.output_image_path}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 