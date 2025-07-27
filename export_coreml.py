import torch
import coremltools as ct
from torchvision import models

# 加载 PyTorch 模型
model = torch.load('chicken_duck_model.pth', map_location='cpu')
model.eval()

# 定义输入示例（必须与训练时尺寸一致）
example_input = torch.rand(1, 3, 224, 224)  # [batch, channels, height, width]

# 转换为 Core ML 格式
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, 224, 224))],  # 关键修改
    classifier_config=ct.ClassifierConfig(["鸡", "鸭"]),
    convert_to="mlprogram"
)

# 保存模型
mlmodel.save("ChickenDuckClassifier.mlpackage")