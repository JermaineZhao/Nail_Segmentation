# coco2nail_vis_and_masks.py
# Usage:
#   python coco2nail_vis_and_masks.py \
#       --ann _annotations.coco.json \
#       --images ./images \
#       --out_vis ./vis \
#       --out_masks ./masks \
#       --save_instance_masks

import argparse, os, json, cv2, numpy as np
from tqdm import tqdm

# ----- utils -----
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def resolve_image_path(img_rec, images_dir):
    """
    尝试用 file_name；若找不到，再尝试 extra.name
    """
    fname = img_rec.get("file_name")
    p1 = os.path.join(images_dir, fname) if fname else None
    if p1 and os.path.exists(p1):
        return p1
    # 有些导出会放在 extra.name
    extra = img_rec.get("extra", {})
    alt = extra.get("name")
    if alt:
        p2 = os.path.join(images_dir, alt)
        if os.path.exists(p2):
            return p2
    # 再尝试把扩展名互换（jpg↔png）
    if p1:
        root, ext = os.path.splitext(p1)
        for e in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            p_try = root + e
            if os.path.exists(p_try): return p_try
    return None

def polygon_to_np(seg):
    """ seg: [x1,y1,x2,y2,...] -> np.array of shape (N,1,2) for cv2 """
    arr = np.array(seg, dtype=np.float32).reshape(-1, 2)
    return arr.astype(np.int32).reshape(-1, 1, 2)

def draw_poly_and_bbox(vis, polys, bbox, color, alpha=0.35, thickness=2):
    # 画半透明 polygon
    overlay = vis.copy()
    for p in polys:
        cv2.fillPoly(overlay, [p], color)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, dst=vis)
    # 画边界
    for p in polys:
        cv2.polylines(vis, [p], isClosed=True, color=color, thickness=thickness)
    # 画 bbox
    x,y,w,h = bbox
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(vis, (int(x), int(y)), (x2, y2), color, 2)

def rle_encode_binary_mask(mask):
    """可选：如需写回 COCO，可做 RLE，这里不需要"""
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="Path to _annotations.coco.json")
    ap.add_argument("--images", required=True, help="Directory of source images")
    ap.add_argument("--out_vis", default="./vis", help="Output dir for visualizations")
    ap.add_argument("--out_masks", default="./masks", help="Output dir for merged binary masks")
    ap.add_argument("--save_instance_masks", action="store_true", help="Also save per-instance masks")
    args = ap.parse_args()

    ensure_dir(args.out_vis)
    ensure_dir(args.out_masks)

    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建索引：image_id -> image_record；image_id -> list(annotations)
    images = {im["id"]: im for im in coco.get("images", [])}
    ann_by_img = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # 类别名（可用于可视化文字）
    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}

    for img_id, img_rec in tqdm(images.items(), desc="Processing"):
        img_path = resolve_image_path(img_rec, args.images)
        if not img_path:
            print(f"[WARN] Missing image file for id={img_id}, file_name={img_rec.get('file_name')}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # 初始化可视化底图与二值 mask
        vis = img.copy()
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        anns = ann_by_img.get(img_id, [])
        # 每个实例随机一个可视化颜色
        rng = np.random.default_rng(img_id + 2025)

        for idx, ann in enumerate(anns):
            # 1) 解析 segmentation（可能是多个多边形）
            segs = ann.get("segmentation", [])
            polys = []
            for seg in segs:
                if len(seg) >= 6:  # 至少三点
                    polys.append(polygon_to_np(seg))

            # 2) 画可视化 polygon + bbox
            color = tuple(int(c) for c in rng.integers(30, 230, size=3))  # 避免太浅或太深
            draw_poly_and_bbox(vis, polys, ann.get("bbox", [0,0,0,0]), color)

            # 3) 累加到二值合并 mask （255=前景）
            if polys:
                cv2.fillPoly(merged_mask, polys, 255)

            # 可选：保存“逐实例”掩码
            if args.save_instance_masks:
                inst_mask = np.zeros((h, w), dtype=np.uint8)
                if polys:
                    cv2.fillPoly(inst_mask, polys, 255)
                base = os.path.splitext(os.path.basename(img_path))[0]
                cv2.imwrite(os.path.join(args.out_masks, f"{base}_inst{idx:02d}.png"), inst_mask)

        # 保存可视化图与合并掩码
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.out_vis, f"{base}.jpg"), vis)
        cv2.imwrite(os.path.join(args.out_masks, f"{base}.png"), merged_mask)

if __name__ == "__main__":
    main()
