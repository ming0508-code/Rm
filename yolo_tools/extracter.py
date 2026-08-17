import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

#   在这里修改所有配置（支持绝对路径）

# ---- 输入视频 ----
VIDEO_PATH = "/home/yourname/videos/video1.mp4"   # 视频文件的绝对路径
VIDEO_NUM = 1                                     # 视频序号（用于文件命名前缀）

# ---- 普通输出目录（所有满足帧差的帧） ----
TRAIN_DIR = "/home/yourname/dataset/train/images"   # 训练集图片
VAL_DIR   = "/home/yourname/dataset/val/images"     # 验证集图片

# ---- YOLO 筛选功能（可选） ----
USE_YOLO = True                      # True 启用，False 禁用
MODEL_PATH = "/home/yourname/models/yolov8n.pt"   # 模型绝对路径

# ---- YOLO 筛选后的输出目录（包含图片+标注txt） ----
TRAIN_TEMP_DIR = "/home/yourname/dataset/train_temp"   # 筛选后训练集
VAL_TEMP_DIR   = "/home/yourname/dataset/val_temp"     # 筛选后验证集

# ---- 帧差与跳帧控制 ----
RATIO = 0.8          # 训练集比例（剩余为验证集）
THRESH = 30.0        # 帧差阈值（越大越不敏感）
SKIP_FRAMES = 5      # 保存一帧后至少跳过的帧数

# ============================================================
#  以下代码无需修改
# ============================================================

def main():
    # 创建目录
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    if USE_YOLO:
        os.makedirs(TRAIN_TEMP_DIR, exist_ok=True)
        os.makedirs(VAL_TEMP_DIR, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    split_idx = int(total_frames * RATIO)
    print(f"总帧数: {total_frames}, 分割点(训练/验证): {split_idx}")

    # 加载YOLO模型
    if USE_YOLO:
        print(f"加载YOLO模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        model.overrides["imgsz"] = 640
    else:
        model = None

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("视频为空")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    train_count = 0
    val_count = 0
    train_temp_count = 0
    val_temp_count = 0
    last_saved_idx = -100

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = np.mean(diff)

        # 条件1：帧差 > 阈值 且 跳过足够帧
        if not (mean_diff > THRESH and (frame_count - last_saved_idx) > SKIP_FRAMES):
            prev_gray = gray
            continue

        # 条件2（可选）：YOLO置信度筛选
        if USE_YOLO:
            results = model(frame, verbose=False)
            confs = []
            for r in results:
                if r.boxes is not None:
                    confs.extend(r.boxes.conf.cpu().numpy().tolist())
            high_conf = any(c > 0.8 for c in confs)
            low_conf = any(0.1 <= c <= 0.3 for c in confs)
            if not (high_conf or low_conf):
                prev_gray = gray
                continue

        # 确定属于训练集还是验证集
        is_train = frame_count < split_idx
        base_dir = TRAIN_DIR if is_train else VAL_DIR
        temp_dir = TRAIN_TEMP_DIR if is_train else VAL_TEMP_DIR if USE_YOLO else None

        # 计数器
        if is_train:
            train_count += 1
            counter = train_count
        else:
            val_count += 1
            counter = val_count

        # 保存到普通目录
        img_name = f"{VIDEO_NUM}_{counter:05d}.jpg"
        img_path = os.path.join(base_dir, img_name)
        cv2.imwrite(img_path, frame)

        # 如果启用YOLO，保存到临时目录并生成标注
        if USE_YOLO:
            if is_train:
                train_temp_count += 1
                temp_counter = train_temp_count
            else:
                val_temp_count += 1
                temp_counter = val_temp_count

            temp_img_name = f"{VIDEO_NUM}_{temp_counter:05d}.jpg"
            temp_img_path = os.path.join(temp_dir, temp_img_name)
            cv2.imwrite(temp_img_path, frame)

            # 生成YOLO格式txt标注
            h, w = frame.shape[:2]
            txt_name = f"{VIDEO_NUM}_{temp_counter:05d}.txt"
            txt_path = os.path.join(temp_dir, txt_name)
            with open(txt_path, 'w') as f:
                for r in results:
                    if r.boxes is None:
                        continue
                    boxes = r.boxes.xywhn.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    for box, cls_id, conf in zip(boxes, cls_ids, confs):
                        f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

        # 更新
        prev_gray = gray
        last_saved_idx = frame_count

        print(f"帧 {frame_count} 保存: {'训练' if is_train else '验证'} - {img_name}" + 
              (f" (临时: {temp_img_name})" if USE_YOLO else ""))

        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")

    cap.release()
    print("\n处理完成！")
    print(f"普通训练集: {train_count} 张")
    print(f"普通验证集: {val_count} 张")
    if USE_YOLO:
        print(f"筛选后训练集(临时): {train_temp_count} 张")
        print(f"筛选后验证集(临时): {val_temp_count} 张")

if __name__ == "__main__":
    main()
