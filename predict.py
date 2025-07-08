import torch
from torchvision import models, transforms
from PIL import Image
import os

# ====== 参数配置 ======
image_path = 'test.jpg'         # 要预测的图片
model_path = 'model.pth'        # 模型文件
class_names = ['heavy', 'light_positive', 'moderate', 'negative']  # 你的类别顺序

# ====== 图像预处理（与训练时保持一致） ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== 加载图片 ======
if not os.path.exists(image_path):
    print(f"❌ 找不到测试图像：{image_path}")
    exit()

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度

# ====== 加载模型并修改分类层 ======
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ====== 预测 ======
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    print(f"✅ 预测结果：{predicted_class}")
