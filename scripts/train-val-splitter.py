import os, shutil, random

# Label Studio export 경로
export_dir = "layout_data/kexam"
images_dir = os.path.join(export_dir, "images")
labels_dir = os.path.join(export_dir, "labels")

# 새 YOLO dataset 구조
base_dir = "layout_data/kexam"
splits = ["train", "val"]
for split in splits:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

# 이미지 목록 불러오기
images = [f for f in os.listdir(images_dir) if f.endswith((".jpg",".png"))]
random.shuffle(images)
print("발견한 이미지 수:", len(images))

# train:val = 8:2
split_idx = int(0.8 * len(images))
train_imgs, val_imgs = images[:split_idx], images[split_idx:]
print("분할 예정 - Train:", len(train_imgs), "장,", "Val:", len(val_imgs), "장")

def move_files(img_list, split):
    total = len(img_list)
    processed = 0
    for i, img in enumerate(img_list, 1):
        label = img.rsplit(".",1)[0] + ".txt"
        shutil.copy(os.path.join(images_dir, img), os.path.join(base_dir, "images", split, img))
        shutil.copy(os.path.join(labels_dir, label), os.path.join(base_dir, "labels", split, label))
        processed = i
        if i % 50 == 0 or i == total:
            print(f"[{split}] {processed}/{total} 처리")
    return processed

train_count = move_files(train_imgs, "train")
val_count = move_files(val_imgs, "val")

print(f"처리 완료 - Train: {train_count}장, Val: {val_count}장, 총 {train_count + val_count}장")
print("Dataset prepared at:", base_dir)