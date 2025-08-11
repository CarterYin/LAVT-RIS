# LAVT Demo 使用指南

## 预训练模型

下载预训练模型到 `checkpoints` 目录：

- **RefCOCO**: [下载链接](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing)
- **RefCOCO+**: [下载链接](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing)
- **G-Ref (UMD)**: [下载链接](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing)
- **G-Ref (Google)**: [下载链接](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing)

### 从Google Drive下载到本地
上传多个文件到远程服务器：
```bash
yinchao@yinchaodeMacBook-Air % cd Desktop
yinchao@yinchaodeMacBook-Air Desktop % scp gref_google.pth gref_umd.pth refcoco.pth refcoco+.pth y****@*****:/home/yinchao/LAVT-RIS/checkpoints
```


## 环境要求

```bash
pip install -r requirements.txt
```

```bash
conda create -n lavt python=3.7.16
conda activate lavt
```

## 使用示例

```bash
python lavt_demo.py --input_image_path demo/real_test.jpg --prompt "the cat" --output_image_path demo/result_cat.jpg --device cpu
```

```bash
python lavt_demo.py --input_image_path demo/person_test.jpg --prompt "the man" --output_image_path demo/result_man.jpg --device cpu
```

## 输出说明

程序会生成一个可视化结果图像，其中：
- 红色区域：检测到的目标对象
- 黑色轮廓：对象边界
- 原始图像：作为背景

