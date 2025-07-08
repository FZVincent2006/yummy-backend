import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# 原始图像目录
input_dir = 'dataset'
# 增强后的图像保存目录
output_dir = 'augmented_dataset'

# 定义增强方法组合（每张图增强10张）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((224, 224))
])

# 增强函数主体
def augment_images():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = ImageFolder(input_dir)

    for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
        class_name = dataset.classes[label]
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        for j in range(10):  # 每张图扩充10张
            aug_img = transform(img)
            aug_img.save(os.path.join(save_path, f"{i}_{j}.jpg"))
if __name__ == '__main__':
    print("开始增强...")
    augment_images()
    print("增强完成！")

if __name__ == '__main__':
    augment_images()
