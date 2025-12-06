import os
import json
import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models


# =========================
# 配置
# =========================

TRAIN_DIR = r"C:\Users\Jermaine Zhao\Downloads\Nail_Curve\data\Nail_Curve_Train_Set"
TEST_DIR = r"C:\Users\Jermaine Zhao\Downloads\Nail_Curve\data\Nail_Curve_Test_Set"
IMAGE_SIZE = 720  # 如果你统一 resize 到 720x720，就用这个
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE_HEAD = 1e-3   # 先训 fc 头
LEARNING_RATE_FULL = 1e-4   # 然后微调全模型
FREEZE_EPOCHS = 5           # 先只训练 fc 的 epoch 数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1


# =========================
# 工具函数：读取 LabelMe JSON -> 3 点坐标
# =========================

def load_labelme_points(json_path):
    """
    从 LabelMe JSON 中提取 Left / Middle / Right 三个点的 (x, y)
    并按顺序返回：[Lx, Ly, Mx, My, Rx, Ry]（像素坐标）
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    w = data["imageWidth"]
    h = data["imageHeight"]

    points_dict = {"Left": None, "Middle": None, "Right": None}

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in points_dict:
            continue
        # 每个 point 是 [[x,y]]
        x, y = shape["points"][0]
        points_dict[label] = (x, y)

    # 简单检查
    for k, v in points_dict.items():
        if v is None:
            raise ValueError(f"{json_path} 中缺少标注点: {k}")

    Lx, Ly = points_dict["Left"]
    Mx, My = points_dict["Middle"]
    Rx, Ry = points_dict["Right"]

    return (Lx, Ly, Mx, My, Rx, Ry, w, h)


# =========================
# 自定义 Dataset
# =========================

class NailKeypointDataset(Dataset):
    def __init__(self, root_dir, image_size=IMAGE_SIZE, train=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train

        # 找所有图片文件（你可以根据实际后缀修改）
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(root_dir, ext)))

        # 对应 json 必须存在
        self.samples = []
        for img_path in sorted(image_paths):
            base, _ = os.path.splitext(img_path)
            json_path = base + ".json"
            if os.path.exists(json_path):
                self.samples.append((img_path, json_path))
            else:
                print(f"[Warning] JSON not found for image: {img_path}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid (image, json) pairs found in {root_dir}")

        # 图像级增强：不影响坐标的可以直接用 torchvision
        if self.train:
            self.color_jitter = T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02
            )
        else:
            self.color_jitter = None

        # Resize 变换（图像 + 坐标都要变）
        self.resize = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # 1. 读取图像
        img = Image.open(img_path).convert("RGB")

        # 2. 读取原始关键点（像素坐标）
        Lx, Ly, Mx, My, Rx, Ry, w, h = load_labelme_points(json_path)

        # 把 keypoints 组织成数组，方便处理
        # 顺序: Left(0), Middle(1), Right(2)
        keypoints = torch.tensor([
            [Lx, Ly],
            [Mx, My],
            [Rx, Ry]
        ], dtype=torch.float32)  # shape: (3, 2)

        # 3. Resize 图像 & 同步缩放 keypoints
        orig_w, orig_h = img.size  # 注意 PIL 是 (w, h)
        img = self.resize(img)
        new_w, new_h = self.image_size, self.image_size

        # 等比拉伸是非均匀的，这里假设你已经统一成正方形图，
        # 或者直接按比例缩放 x,y：
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # 4. 数据增强（仅 train）
        if self.train:
            # 4.1 色彩增强（不影响 keypoints）
            img = self.color_jitter(img)

            # 4.2 随机水平翻转（需要同步改变 keypoints）
            if random.random() < 0.5:
                img = TF.hflip(img)
                # x' = new_w - 1 - x
                keypoints[:, 0] = new_w - 1 - keypoints[:, 0]

                # 左右手指交换：Left <-> Right
                # indices: 0 <-> 2
                left = keypoints[0].clone()
                right = keypoints[2].clone()
                keypoints[0] = right
                keypoints[2] = left

        # 5. 转成 [0,1] 归一化坐标
        keypoints_norm = keypoints.clone()
        keypoints_norm[:, 0] /= new_w
        keypoints_norm[:, 1] /= new_h

        # 展平成 6 维向量: [Lx, Ly, Mx, My, Rx, Ry]
        target = keypoints_norm.view(-1)  # shape: (6,)

        # 6. 转 tensor，归一化图像
        img_tensor = TF.to_tensor(img)  # [0,1]
        img_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 均值
            std=[0.229, 0.224, 0.225]
        )(img_tensor)

        return img_tensor, target


# =========================
# 模型定义：ResNet34 + 6维回归头
# =========================

def create_model():
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = resnet34.fc.in_features

    resnet34.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),  # 新增
        nn.Linear(128, 6),
        nn.Sigmoid()  # 输出到 [0,1]，对应归一化坐标
    )
    return resnet34


# =========================
# 训练 & 验证循环
# =========================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)  # (B, 6)

        optimizer.zero_grad()
        outputs = model(images)      # (B, 6)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# =========================
# 主函数
# =========================

def main():
    print(f"Using device: {DEVICE}")

    # 数据集 & DataLoader
    train_dataset = NailKeypointDataset(TRAIN_DIR, image_size=IMAGE_SIZE, train=True)
    test_dataset = NailKeypointDataset(TEST_DIR, image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型
    model = create_model().to(DEVICE)

    # 损失函数
    criterion = nn.MSELoss()

    # ========== 第一阶段：只训练 fc 头 ==========
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=LEARNING_RATE_HEAD)

    best_val_loss = float("inf")

    print("=== Stage 1: Train head only ===")
    for epoch in range(1, FREEZE_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer_head, criterion, DEVICE)
        val_loss = eval_one_epoch(model, test_loader, criterion, DEVICE)

        if epoch % PRINT_EVERY == 0:
            print(f"[Stage1][Epoch {epoch}/{FREEZE_EPOCHS}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_dropout.pth")
            print(f"  --> New best model saved (val_loss={val_loss:.6f})")

    # ========== 第二阶段：微调整个模型 ==========
    for param in model.parameters():
        param.requires_grad = True

    # optimizer_full = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FULL)
    optimizer_full = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE_FULL,
        weight_decay=1e-4
    )


    print("=== Stage 2: Fine-tune full model ===")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer_full, criterion, DEVICE)
        val_loss = eval_one_epoch(model, test_loader, criterion, DEVICE)

        if epoch % PRINT_EVERY == 0:
            print(f"[Stage2][Epoch {epoch}/{NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_dropout.pth")
            print(f"  --> New best model saved (val_loss={val_loss:.6f})")

    print("Training finished. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
