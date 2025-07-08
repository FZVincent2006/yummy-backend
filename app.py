from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# ======= 配置模型路径和类别名 =======
model_path = 'model.pth'
class_names = ['heavy', 'light_positive', 'moderate', 'negative']

# ======= 每个类别的中文建议 =======
suggestions = {
    'negative': '安全请放心使用',
    'light_positive': '请将去除剂倒入油壶，静置30分钟后过滤',
    'moderate': '请将去除剂倒入油壶，静置30分钟后过滤',
    'heavy': '黄曲霉素严重超标，请勿再继续食用'
}

# ======= 图像预处理方式（与训练一致） =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======= 加载模型结构和参数 =======
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ======= 接口：上传图片并返回预测等级和建议 =======
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        suggestion = suggestions[label]

    return jsonify({
        'prediction': label,
        'suggestion': suggestion
    })

# ======= 启动 Flask 服务 =======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
