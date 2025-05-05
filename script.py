import os
import shutil
from glob import glob

def build_dataset(groundtruth_root, image_root, output_root):
    rgb_root = os.path.join(output_root, "raw_rgb")
    label_root = os.path.join(output_root, "label")
    os.makedirs(rgb_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)

    scenes = sorted(os.listdir(groundtruth_root))
    for scene in scenes:
        gt_img_dir = os.path.join(groundtruth_root, scene, "image_03")
        rgb_img_dir = os.path.join(image_root, scene, "data")

        if not os.path.isdir(gt_img_dir) or not os.path.isdir(rgb_img_dir):
            print(f"[跳过] 缺失目录: {scene}")
            continue

        output_rgb_scene = os.path.join(rgb_root, scene)
        output_label_scene = os.path.join(label_root, scene)
        os.makedirs(output_rgb_scene, exist_ok=True)
        os.makedirs(output_label_scene, exist_ok=True)

        # 以深度图为准
        for gt_path in sorted(glob(os.path.join(gt_img_dir, "*.png"))):
            fname = os.path.basename(gt_path)
            rgb_path = os.path.join(rgb_img_dir, fname)

            if os.path.exists(rgb_path):
                shutil.copyfile(rgb_path, os.path.join(output_rgb_scene, fname))
                shutil.copyfile(gt_path, os.path.join(output_label_scene, fname))
            else:
                print(f"[缺失 RGB] {scene}/{fname}，跳过")

    print("✅ 数据集构建完成")

# 示例使用
build_dataset(
    groundtruth_root="dataset/groundtruth",
    image_root="dataset/image",
    output_root="dataset"
)
