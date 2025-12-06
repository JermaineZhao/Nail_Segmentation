import cv2
import numpy as np

def process_image(image_path, output_path="output_with_curve.png",
                  show_window=False):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h_img, w_img = img.shape[:2]

    # 2. 颜色分割：提取绿色 C curve
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 下面的阈值适合“荧光绿”，你可以按实际图片微调
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 可选：去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. 提取所有绿色像素的 (x, y)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("No green pixels found. Adjust HSV thresholds.")

    points = np.stack([xs, ys], axis=1).astype(np.float32)  # N x 2, [x, y]

    # 4. 找 L / R / T
    # L: x 最小；R: x 最大；T: y 最小（图像坐标里最上方）
    idx_L = np.argmin(points[:, 0])
    idx_R = np.argmax(points[:, 0])
    idx_T = np.argmin(points[:, 1])

    L = points[idx_L]  # [x_L, y_L]
    R = points[idx_R]  # [x_R, y_R]
    T = points[idx_T]  # [x_T, y_T]

    # 5. 用多项式拟合 C curve：y = ax^2 + bx + c
    x = points[:, 0]
    y = points[:, 1]

    # 为了减少垂直噪声的影响，可以对同一个 x 取平均 y（可选）
    # 这里简单直接用所有点 polyfit 二次曲线
    poly_deg = 2
    coeffs = np.polyfit(x, y, poly_deg)  # [a, b, c]
    a, b, c = coeffs

    # 6. 在 L、R 之间均匀采样 x，画出拟合曲线（蓝色）
    x_min = int(np.min(x))
    x_max = int(np.max(x))
    num_samples = 300
    xs_fit = np.linspace(x_min, x_max, num_samples)
    ys_fit = a * xs_fit**2 + b * xs_fit + c

    # 把拟合曲线画到图片上
    img_with_curve = img.copy()
    for i in range(num_samples - 1):
        x1, y1 = int(xs_fit[i]),   int(ys_fit[i])
        x2, y2 = int(xs_fit[i+1]), int(ys_fit[i+1])

        # 防止越界
        if 0 <= x1 < w_img and 0 <= y1 < h_img and \
           0 <= x2 < w_img and 0 <= y2 < h_img:
            # 画蓝色线条 (B, G, R) = (255, 0, 0)
            cv2.line(img_with_curve, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 把 L / R / T 三个点用红点标出来（辅助查看）
    for P in [L, R, T]:
        cv2.circle(img_with_curve, (int(P[0]), int(P[1])), 4, (0, 0, 255), -1)

    # 7. 计算 width / height / C（几何方式）
    v = R - L                           # chord LR
    width = float(np.linalg.norm(v))

    w_vec = T - L
    cross = v[0] * w_vec[1] - v[1] * w_vec[0]
    height = float(abs(cross) / np.linalg.norm(v))

    C = width / height if height > 1e-6 else float("inf")

    print(f"width  (pixels): {width:.4f}")
    print(f"height (pixels): {height:.4f}")
    print(f"C = width/height: {C:.4f}")

    # 8. 把结果写在图片上
    text1 = f"w={width:.1f}px"
    text2 = f"h={height:.1f}px"
    text3 = f"C={C:.2f}"

    cv2.putText(img_with_curve, text1, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img_with_curve, text2, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img_with_curve, text3, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 9. 保存结果
    cv2.imwrite(output_path, img_with_curve)
    print(f"Saved output to: {output_path}")

    # 可选：弹窗显示
    if show_window:
        cv2.imshow("mask", mask)
        cv2.imshow("fitted_curve", img_with_curve)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 也可以把数值返回出去
    return width, height, C

if __name__ == "__main__":
    # 替换成你的图片路径
    image_path = "nail.jpg"
    output_path = "nail_curve_result.jpg"
    process_image(image_path, output_path, show_window=False)