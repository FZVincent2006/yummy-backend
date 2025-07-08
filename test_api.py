import requests

url = 'https://yummy-backend-uncs.onrender.com/predict'
image_path = r'D:\testpic\1.jpg'

try:
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print("状态码：", response.status_code)

    try:
        print("返回内容：", response.json())
    except Exception as e:
        print("无法解析返回的 JSON，原始响应内容：", response.text)

except FileNotFoundError:
    print("❌ 找不到图片文件，请检查路径是否正确！")
except Exception as e:
    print("❌ 请求出错：", str(e))
