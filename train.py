import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def main():
    # ======== 配置参数 ========
    data_dir = 'augmented_dataset'  # 数据文件夹
    num_classes = 4
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    model_save_path = 'model.pth'

    # ======== 图像预处理 ========
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ======== 加载数据集 ========
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    print("检测类别：", class_names)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ======== 加载预训练模型 ========
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ======== 开始训练 ========
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

        accuracy = correct.double() / len(dataset)
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Accuracy={accuracy:.4f}")

    # ======== 保存模型 ========
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ 模型已保存为 {model_save_path}")

# ✅ 正确的执行入口
if __name__ == '__main__':
    print("🔥 开始训练模型...")
    main()

