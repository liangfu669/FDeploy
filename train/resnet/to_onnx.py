import torch
import torchvision.models as models

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)

# 将模型设置为评估模式
model.eval()

# 定义输入张量，需要与模型的输入张量形状相同
input_shape = (1, 3, 224, 224)
x = torch.randn(input_shape)


# 需要指定输入张量，输出文件路径和运行设备
# 默认情况下，输出张量的名称将基于模型中的名称自动分配
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将 PyTorch 模型转换为 ONNX 格式
output_file = "resnet18.onnx"
torch.onnx.export(model.to(device), x.to(device), output_file, export_params=True)