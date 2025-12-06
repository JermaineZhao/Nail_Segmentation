# python test.py --input_dir "C:\Users\Jermaine Zhao\Downloads\Nail_Curve\data\Nail_Curve_Test_Set" --output_dir "C:\Users\Jermaine Zhao\Downloads\Nail_Curve\Inference_dropout" --model_path best_model_dropout.pth

import os
import argparse
from glob import glob
import json
import math

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models


IMAGE_SIZE = 720   # 要和训练时保持一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 创建模型（要和训练时一模一样）
# =========================
def create_model():
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = resnet34.fc.in_features

    resnet34.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),  # 和 train.py 保持一致
        nn.Linear(128, 6),
        nn.Sigmoid()       # 输出 [0,1] 归一化坐标
    )
    return resnet34


# =========================
# 从 LabelMe JSON 读取 GT 三个点（原图像素坐标）
# =========================
def load_labelme_points_orig(json_path):
    """
    返回一个 dict:
    {
        "Left":  (x, y),
        "Middle": (x, y),
        "Right": (x, y)
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    pts = {"Left": None, "Middle": None, "Right": None}

    for shape in data["shapes"]:
        label = shape["label"]
        if label in pts:
            x, y = shape["points"][0]
            pts[label] = (x, y)

    for k, v in pts.items():
        if v is None:
            raise ValueError(f"{json_path} 中缺少标注点: {k}")

    return pts


# =========================
# 处理单张图片并预测 keypoints（返回原图坐标）
# =========================
def predict_keypoints_for_image(model, img_path, output_dir=None, draw=True):
    # 1. 读取图像
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # 2. Resize 到 720x720
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
    img_resized = resize(img)

    # 3. 转 tensor + normalize
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 4. 推理
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)  # (1,6)

    preds = preds[0].cpu().numpy()  # 6 维： [Lx, Ly, Mx, My, Rx, Ry]

    # 5. 恢复到 720x720 像素坐标
    Lx_norm, Ly_norm, Mx_norm, My_norm, Rx_norm, Ry_norm = preds
    Lx = Lx_norm * IMAGE_SIZE
    Ly = Ly_norm * IMAGE_SIZE
    Mx = Mx_norm * IMAGE_SIZE
    My = My_norm * IMAGE_SIZE
    Rx = Rx_norm * IMAGE_SIZE
    Ry = Ry_norm * IMAGE_SIZE

    # 6. 反缩放到原始尺寸
    scale_x = orig_w / IMAGE_SIZE
    scale_y = orig_h / IMAGE_SIZE

    Lx_orig = Lx * scale_x
    Ly_orig = Ly * scale_y
    Mx_orig = Mx * scale_x
    My_orig = My * scale_y
    Rx_orig = Rx * scale_x
    Ry_orig = Ry * scale_y

    # 7. 可视化：在原始图上画出3个点
    if draw and output_dir is not None:
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        r = max(3, int(min(orig_w, orig_h) * 0.01))  # 圆半径，跟尺寸稍微挂钩

        # 画圆 + 标 label
        def draw_point(x, y, color, text):
            x0, y0 = x - r, y - r
            x1, y1 = x + r, y + r
            draw.ellipse([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x + r + 2, y - r - 2), text, fill=color)

        draw_point(Lx_orig, Ly_orig, "red", "Left")
        draw_point(Mx_orig, My_orig, "yellow", "Middle")
        draw_point(Rx_orig, Ry_orig, "blue", "Right")

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, base_name)
        draw_img.save(save_path)
        print(f"Saved visualized result to: {save_path}")

    # 8. 返回坐标（原图坐标系）
    return {
        "image": img_path,
        "Left":  (Lx_orig, Ly_orig),
        "Middle": (Mx_orig, My_orig),
        "Right": (Rx_orig, Ry_orig),
    }


# =========================
# 主函数：支持单张 or 目录 + 计算平均误差
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="单张图片路径"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="包含多张图片的输入目录（会尝试读取同名 .json 做误差计算）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="输出可视化图片目录"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="训练好的模型权重路径"
    )

    args = parser.parse_args()

    if args.image is None and args.input_dir is None:
        print("必须指定 --image 或 --input_dir 其中一个")
        return

    # 1. 创建模型 & 加载权重
    model = create_model().to(DEVICE)
    state_dict = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {args.model_path}")

    # 如果是单张图，就只画点 & 打印预测坐标，不算平均误差
    if args.image is not None:
        result = predict_keypoints_for_image(
            model,
            args.image,
            args.output_dir,
            draw=True
        )
        print("Predicted keypoints (original image coordinates):")
        for k, v in result.items():
            if k == "image":
                continue
            print(f"  {k}: (x={v[0]:.1f}, y={v[1]:.1f})")
        return

    # 2. 如果是整个目录：逐张推理 + 计算误差
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images in {args.input_dir}")

    # 误差累积
    sum_err_left = 0.0
    sum_err_mid = 0.0
    sum_err_right = 0.0
    count = 0

    for img_path in image_paths:
        # 推理预测点
        pred = predict_keypoints_for_image(
            model,
            img_path,
            args.output_dir,
            draw=True
        )

        # 找到对应的 json
        base, _ = os.path.splitext(img_path)
        json_path = base + ".json"
        if not os.path.exists(json_path):
            print(f"[Warning] JSON not found for image: {img_path}, skip error calc.")
            continue

        # 读取 GT
        gt_pts = load_labelme_points_orig(json_path)

        # 计算三个点的像素误差（欧氏距离）
        def dist(p_pred, p_gt):
            return math.sqrt((p_pred[0] - p_gt[0])**2 + (p_pred[1] - p_gt[1])**2)

        err_left = dist(pred["Left"],   gt_pts["Left"])
        err_mid  = dist(pred["Middle"], gt_pts["Middle"])
        err_right= dist(pred["Right"],  gt_pts["Right"])

        sum_err_left  += err_left
        sum_err_mid   += err_mid
        sum_err_right += err_right
        count += 1

    if count == 0:
        print("没有成功计算误差的样本（可能没有匹配到任何 json）")
        return

    avg_left  = sum_err_left  / count
    avg_mid   = sum_err_mid   / count
    avg_right = sum_err_right / count
    avg_all   = (sum_err_left + sum_err_mid + sum_err_right) / (count * 3.0)

    print("====================================")
    print(f"共评估样本数: {count}")
    print(f"Left   平均误差: {avg_left:.2f} px")
    print(f"Middle 平均误差: {avg_mid:.2f} px")
    print(f"Right  平均误差: {avg_right:.2f} px")
    print("------------------------------------")
    print(f"整体 3 点平均误差: {avg_all:.2f} px")
    print("====================================")


if __name__ == "__main__":
    main()
