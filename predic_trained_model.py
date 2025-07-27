import argparse
import torch
from torchvision import transforms
from PIL import Image


def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def predict_image(model, image_path):
    """预测单张图片"""
    # 预处理（必须与训练时一致！）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return "鸡" if pred.item() == 0 else "鸭"


if __name__ == '__main__':
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser(description='鸡鸭分类器')
    parser.add_argument('--image', type=str, required=True, help='待预测图片路径')
    parser.add_argument('--model', type=str, default='chicken_duck_model.pth',
                        help='模型路径（默认: chicken_duck_model.pth）')
    args = parser.parse_args()

    # 2. 加载模型并预测
    model = load_model(args.model)
    result = predict_image(model, args.image)

    # 3. 打印结果
    print(f"预测结果: {result}")