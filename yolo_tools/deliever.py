import os
import shutil
from pathlib import Path

#  配置区域（使用绝对路径）

# ---- 源目录 ----
SRC_PATH = "/home/yourname/raw_data"   # 原始数据所在文件夹

# ---- 移动类型与目标类型 ----
MOVE_TYPE = "picture"   # 可选 "picture" 或 "txt"
AIM_TYPE = "train"      # 可选 "train" 或 "val"

# ---- 四个目标路径（根据 MOVE_TYPE 和 AIM_TYPE 选择使用） ----
TRAIN_IMG_PATH = "/home/yourname/dataset/train/images"
TRAIN_LABEL_PATH = "/home/yourname/dataset/train/labels"
VAL_IMG_PATH = "/home/yourname/dataset/val/images"
VAL_LABEL_PATH = "/home/yourname/dataset/val/labels"

# ---- 其他选项 ----
RECURSIVE = True       # 是否递归搜索子目录
OVERWRITE = False      # 目标文件已存在时是否覆盖
DRY_RUN = False        # 是否模拟运行

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def get_destination_path():
    """根据 MOVE_TYPE 和 AIM_TYPE 选择目标路径"""
    if MOVE_TYPE == "picture":
        if AIM_TYPE == "train":
            return Path(TRAIN_IMG_PATH)
        elif AIM_TYPE == "val":
            return Path(VAL_IMG_PATH)
    elif MOVE_TYPE == "txt":
        if AIM_TYPE == "train":
            return Path(TRAIN_LABEL_PATH)
        elif AIM_TYPE == "val":
            return Path(VAL_LABEL_PATH)
    raise ValueError(f"无效的 MOVE_TYPE={MOVE_TYPE} 或 AIM_TYPE={AIM_TYPE}")

def move_files(src_dir, dst_dir, extensions, recursive, overwrite, dry_run):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    
    if not src_path.exists():
        print(f"错误：源文件夹不存在 - {src_path}")
        return 0, 0

    dst_path.mkdir(parents=True, exist_ok=True)

    if recursive:
        file_iter = src_path.rglob("*")
    else:
        file_iter = src_path.glob("*")

    files_to_move = [f for f in file_iter if f.is_file() and f.suffix.lower() in extensions]
    
    if not files_to_move:
        print(f"在 {src_path} 中未找到 {extensions} 类型的文件")
        return 0, 0

    moved_count = 0
    skipped_count = 0

    for src_file in files_to_move:
        dest_file = dst_path / src_file.name

        if dest_file.exists() and not overwrite:
            print(f"跳过 {src_file.name}（目标已存在）")
            skipped_count += 1
            continue

        if dry_run:
            print(f"[模拟] 移动: {src_file} -> {dest_file}")
        else:
            shutil.move(str(src_file), str(dest_file))
            print(f"移动: {src_file} -> {dest_file}")
        moved_count += 1

    return moved_count, skipped_count


def main():
    # 确定扩展名
    if MOVE_TYPE == "picture":
        extensions = IMAGE_EXTENSIONS
    else:  # txt
        extensions = {'.txt'}

    dst_path = get_destination_path()
    print(f"源目录: {SRC_PATH}")
    print(f"目标目录: {dst_path}")
    print(f"移动类型: {MOVE_TYPE} ({', '.join(extensions)})")
    print(f"递归搜索: {'是' if RECURSIVE else '否'}")
    print(f"覆盖模式: {'是' if OVERWRITE else '否（跳过）'}")
    print(f"模拟运行: {'是' if DRY_RUN else '否'}")
    print()

    moved, skipped = move_files(
        SRC_PATH,
        dst_path,
        extensions,
        RECURSIVE,
        OVERWRITE,
        DRY_RUN
    )

    print(f"\n完成！移动 {moved} 个文件，跳过 {skipped} 个。")


if __name__ == "__main__":
    main()
