import os
import cv2
from tqdm import tqdm

input_folder = "Raw_Nail_Curve_Test_Set"
output_folder = "Nail_Curve_Test_Set"
target_size = 720

os.makedirs(output_folder, exist_ok=True)

def resize_with_padding(img, size=720):
    h, w = img.shape[:2]

    # 计算缩放比例
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建720x720画布（黑色或白色都可以，这里用白色更接近指甲背景）
    canvas = 255 * np.ones((size, size, 3), dtype=np.uint8)

    # 将缩放后的图像放到中心
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


import numpy as np

for file in tqdm(os.listdir(input_folder)):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print("跳过无法读取的文件：", file)
            continue

        output_img = resize_with_padding(img, target_size)

        save_path = os.path.join(output_folder, file)
        cv2.imwrite(save_path, output_img)

print("全部处理完成！已输出到：", output_folder)