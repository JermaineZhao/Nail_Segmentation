# train_nail_coco.py
# 完整训练：COCO(polygon) -> 512x512 在线resize -> U-Net(ResNet34)
# 依赖：pip install segmentation-models-pytorch albumentations opencv-python timm tqdm
# python .\_aaa_train_nail_coco.py --ann "_annotations.coco.json" --images "." --size 512 --num_samples 100 --epochs 60 --batch 8 --out "runs/nail512"


import os, json, random, argparse, time, math, glob
import numpy as np, cv2, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter


# ---------------- utils ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def resolve_image_path(img_rec, images_dir):
    """在 images_dir 中寻找 file_name 或 extra.name；找不到就递归尝试同名不同扩展。"""
    fn = img_rec.get("file_name")
    cand = [fn]
    extra = img_rec.get("extra") or {}
    if extra.get("name"): cand.append(extra["name"])
    # 直接拼路径
    for c in cand:
        if not c: continue
        p = os.path.join(images_dir, c)
        if os.path.exists(p): return p
    # 尝试不同扩展
    bases = []
    for c in cand:
        if not c: continue
        bases.append(os.path.splitext(os.path.basename(c))[0])
    for base in bases:
        for ext in (".jpg",".png",".jpeg",".JPG",".PNG",".JPEG"):
            p = os.path.join(images_dir, base+ext)
            if os.path.exists(p): return p
    # 递归搜
    for c in cand:
        if not c: continue
        name = os.path.basename(c)
        hits = glob.glob(os.path.join(images_dir, "**", name), recursive=True)
        if hits: return hits[0]
    for base in bases:
        for ext in (".jpg",".png",".jpeg",".JPG",".PNG",".JPEG"):
            hits = glob.glob(os.path.join(images_dir, "**", base+ext), recursive=True)
            if hits: return hits[0]
    return None

def polygons_to_mask(polys, h, w):
    """polys: list of [x1,y1,...]; output uint8 mask {0,255}"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv_polys = []
    for seg in polys:
        if len(seg) < 6: 
            continue
        arr = np.array(seg, dtype=np.float32).reshape(-1,2).astype(np.int32)
        cv_polys.append(arr)
    if cv_polys:
        cv2.fillPoly(mask, cv_polys, 255)
    return mask

def dice_coef(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def iou_coef(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    uni = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (uni + eps)).mean()

# -------------- Dataset ---------------
class CocoSegDataset(Dataset):
    def __init__(self, images_dir, images_list, ann_by_img, size=512, is_train=True):
        self.images_dir = images_dir
        self.images_list = images_list  # list of dict (coco image record)
        self.ann_by_img = ann_by_img
        if is_train:
            self.tf = A.Compose([
                A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.6),
                A.ColorJitter(0.2,0.2,0.2,0.05, p=0.3),
                A.Normalize(), ToTensorV2()
            ])
        else:
            self.tf = A.Compose([A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
                                 A.Normalize(), ToTensorV2()])

    def __len__(self): return len(self.images_list)

    def __getitem__(self, idx):
        imrec = self.images_list[idx]
        path = resolve_image_path(imrec, self.images_dir)
        if path is None:
            raise FileNotFoundError(f"Image not found for id={imrec['id']}, file_name={imrec.get('file_name')}")
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(f"Fail to read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # build mask from polygons
        anns = self.ann_by_img.get(imrec["id"], [])
        polys = [seg for ann in anns for seg in ann.get("segmentation", [])]
        mask = polygons_to_mask(polys, h, w)  # 0/255
        mask = (mask > 127).astype(np.float32)

        out = self.tf(image=img, mask=mask)
        x = out["image"]
        y = out["mask"].unsqueeze(0)  # [1,H,W]
        return x, y, os.path.basename(path)

# -------------- Training --------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    vis_dir = ensure_dir(os.path.join(args.out, "vis"))

    # === TensorBoard writer ===
    tb_dir = os.path.join(args.out, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    # load coco
    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    anns = coco["annotations"]

    # group annotations by image_id；只保留有 annotation 的图片（更快看到效果）
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    has_ann_ids = set(ann_by_img.keys())
    images = [im for im in images if im["id"] in has_ann_ids]

    # 随机抽样 num_samples
    random.shuffle(images)
    images = images[:args.num_samples]

    # train/val split（9:1，按 image 级）
    n = len(images)
    val_n = max(20, int(0.1 * n))
    train_imgs = images[:-val_n]
    val_imgs = images[-val_n:]

    print(f"Samples: train={len(train_imgs)}  val={len(val_imgs)}  (total picked={n})")

    # datasets / loaders
    train_ds = CocoSegDataset(args.images, train_imgs, ann_by_img, size=args.size, is_train=True)
    val_ds   = CocoSegDataset(args.images, val_imgs,   ann_by_img, size=args.size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1,args.batch//2), shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    # model / loss / opt
    model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1, activation=None)
    model = model.to(device)
    loss_dice = smp.losses.DiceLoss(mode="binary")
    loss_bce  = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_dice = -1.0
    patience = 12
    bad = 0
    start = time.time()

    for ep in range(1, args.epochs+1):
        model.train()
        tloss, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs} - train")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(x)
                loss = 0.5*loss_dice(logits, y) + 0.5*loss_bce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tloss += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{tloss/steps:.4f}")
        scheduler.step()

        # ---- validation ----
        model.eval()
        vd, viou, vsteps = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="val"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vd += dice_coef(logits, y).item()
                viou += iou_coef(logits, y).item()
                vsteps += 1
        vd /= max(1,vsteps); viou /= max(1,vsteps)
        print(f"[Val] Dice={vd:.4f}  IoU={viou:.4f}  (train_loss_avg={tloss/max(1,steps):.4f})")
        
        # TensorBoard scalars
        writer.add_scalar("train/loss", tloss/max(1,steps), ep)
        writer.add_scalar("val/dice", vd, ep)
        writer.add_scalar("val/iou",  viou, ep)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], ep)


        # ---- save best & visualizations ----
        if vd > best_dice:
            best_dice = vd; bad = 0
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            # 可视化若干张
            save_visuals(model, val_ds, device, vis_dir, ep, max_samples=6,writer=writer)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}.")
                break

    writer.close()
    print(f"TensorBoard logs -> {tb_dir}")
    print(f"Done. Best Dice={best_dice:.4f}. Time={(time.time()-start)/60:.1f} min")
    print(f"Best weights -> {os.path.join(args.out, 'best.pt')}")
    print(f"Visualizations -> {vis_dir}")

def save_visuals(model, dataset, device, vis_dir, epoch, max_samples=6, writer=None):
    model.eval()
    idxs = np.linspace(0, len(dataset)-1, num=min(max_samples, len(dataset)), dtype=int)
    overlay_tensors = []  # for TensorBoard
    for i in idxs:
        x, y, name = dataset[i]
        x1 = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pr = torch.sigmoid(model(x1))[0,0].cpu().numpy()
            
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = x.permute(1,2,0).cpu().numpy()
        img = (img * std + mean)            # 反归一化到 [0,1]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = (pr>0.5).astype(np.uint8)*255
        overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        outp = os.path.join(vis_dir, f"ep{epoch:03d}_{os.path.splitext(name)[0]}.jpg")
        cv2.imwrite(outp, overlay)

        # —— 为 TensorBoard 准备 RGB-CHW tensor (0-1)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_t = torch.from_numpy(overlay_rgb).permute(2,0,1).float()/255.0
        overlay_tensors.append(overlay_t)

    if writer is not None and len(overlay_tensors) > 0:
        grid = torch.stack(overlay_tensors, dim=0)  # [N,3,H,W]
        writer.add_images("val/overlay_best", grid, global_step=epoch)

# -------------- main -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO json path")
    ap.add_argument("--images", required=True, help="dir holding images")
    ap.add_argument("--out", default="runs/nail512", help="output dir")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    train(args)
