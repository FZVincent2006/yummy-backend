from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import models, transforms
import io
import os

app = Flask(__name__)
CORS(app)  # 允许跨域访问（重要）

# ====== 参数配置 ======
model_path = 'model.pth'
class_names = ['heavy', 'light_positive', 'moderate', 'negative']

# ====== 图像预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ====== 加载模型 ======
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return 'Yummy Backend is running!'

@app.route('/predict', methods=['POST'])
def predict():
    print("收到请求，request.files内容：", request.files)
    if 'file' not in request.files:
        print("没有收到file字段")
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['file']
    print("收到file字段，文件名：", file.filename)
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        print("图片处理失败：", str(e))
        return jsonify({'error': 'Invalid image file'}), 400
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    result_map = {
        'negative': '安全请放心使用',
        'light_positive': '请将去除剂倒入油壶，静置30分钟后过滤',
        'moderate': '请将去除剂倒入油壶，静置30分钟后过滤',
        'heavy': '黄曲霉素严重超标，请勿再继续食用'
    }
    return jsonify({
        'class': predicted_class,
        'result': result_map.get(predicted_class, '无反馈内容')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
