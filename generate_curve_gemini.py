# from google import genai
# from google.genai import types
# from PIL import Image

# client = genai.Client(api_key="AIzaSyBAQv45mS8_egLuVNMNwolEZvDpzN0L2lI")

# prompt = (
#     "这是我的指甲从前面的视角看的样子。在原图的基础之上，给我找出并画出指甲的c-curve，并且在每张图上标 3 个点：1）L：自由缘左侧边缘 2）R：自由缘右侧边缘 3）T：自由缘最凸出的那一点（肉眼看是“最向外顶”的位置）。注意：相机位于指甲前方正对指甲前缘，我要标注指甲最前缘的横截面曲线，而不是和手指连接的那部分。原图层变成灰色，你画的curve那一层用绿色。不要用文字标记出L，R，T，画出点和弧线即可。"
# )

# image = Image.open(r"C:\Users\Jermaine Zhao\Downloads\Nail_Curve\data\Nail_Curve_Test_Set\IMG_3653.JPG")

# response = client.models.generate_content(
#     model="gemini-3-pro-image-preview",
#     contents=[prompt, image],
# )

# for part in response.parts:
#     if part.text is not None:
#         print(part.text)
#     elif part.inline_data is not None:
#         image = part.as_image()
#         image.save("generated_image.png")


import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ================== 配置区域 ==================

API_KEY = "AIzaSyBAQv45mS8_egLuVNMNwolEZvDpzN0L2lI"  # 建议改成 os.environ["GEMINI_API_KEY"]

INPUT_DIR = r"C:\Users\Jermaine Zhao\Downloads\Nail_Curve\data\Nail_Curve_Test_Set"
OUTPUT_DIR = r"C:\Users\Jermaine Zhao\Downloads\Nail_Curve\Inference_curve_banana"

# 每次请求之间停顿的秒数，避免瞬间打太多请求
SLEEP_SECONDS = 5.0

# 允许处理的图片后缀
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 生成图片的提示词（保持你原来的语义）
PROMPT = (
    "这是我的指甲从前面的视角看的样子。在原图的基础之上，给我找出并画出指甲的 c-curve，"
    "并且在图上标 3 个点：1）L：自由缘左侧边缘 2）R：自由缘右侧边缘 "
    "3）T：自由缘最凸出的那一点（肉眼看是“最向外顶”的位置）。"
    "注意：相机位于指甲前方正对指甲前缘，我要标注指甲最前缘的横截面曲线，"
    "而不是和手指连接的那部分。原图层变成灰色，你画的 curve 那一层用绿色。"
    "不要用文字标记 L、R、T，只画出三个点和绿色弧线即可。"
)

# ================== 主逻辑 ==================

def main():
    client = genai.Client(api_key=API_KEY)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有图片文件
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )

    print(f"发现 {len(image_paths)} 张图片，开始处理……")

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] 处理中：{img_path.name}")

        try:
            # 读取原始图片
            image = Image.open(img_path)

            # 调用模型（单张，顺序处理，避免并发过多）
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[PROMPT, image],
            )

            # 有些返回里会同时带 text 和 image，这里只关心图片部分
            saved_any = False
            for part_i, part in enumerate(response.parts):
                if part.inline_data is not None:
                    out_image = part.as_image()

                    # 文件名：原文件名_styled.png（或直接覆盖后缀）
                    out_name = f"{img_path.stem}_curve{part_i}.png"
                    out_path = output_dir / out_name
                    out_image.save(out_path)
                    saved_any = True

            if not saved_any:
                print(f"  ⚠ 没有在返回中找到图片数据，可能只返回了文本。")

        except Exception as e:
            print(f"  ❌ 处理 {img_path.name} 时出错：{e}")

        # 为了避免 API 短时间请求过多，适当 sleep
        if idx < len(image_paths):
            time.sleep(SLEEP_SECONDS)

    print("全部处理完成！")

if __name__ == "__main__":
    main()
