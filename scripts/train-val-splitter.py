import os
import shutil
import random
import argparse
from typing import List, Tuple


def list_images(images_dir: str, extensions: Tuple[str, ...]) -> List[str]:
    exts = tuple(e.lower() for e in extensions)
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(exts)]
    return sorted(files)


def ensure_dest_dirs(base_dir: str, splits: Tuple[str, str] = ("train", "val")) -> None:
    for split in splits:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)


def clear_dest_dirs(base_dir: str, splits: Tuple[str, str] = ("train", "val")) -> None:
    for split in splits:
        for sub in ("images", "labels"):
            d = os.path.join(base_dir, sub, split)
            if os.path.isdir(d):
                for name in os.listdir(d):
                    try:
                        os.remove(os.path.join(d, name))
                    except Exception:
                        pass


def copy_or_move(src: str, dst: str, do_move: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if do_move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def split_dataset(
    export_dir: str,
    base_dir: str,
    train_ratio: float,
    extensions: Tuple[str, ...],
    seed: int,
    do_move: bool,
    clear_dest: bool,
    skip_missing_labels: bool,
    dry_run: bool,
) -> Tuple[int, int, int]:
    images_dir = os.path.join(export_dir, "images")
    labels_dir = os.path.join(export_dir, "labels")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"이미지/라벨 디렉터리를 찾을 수 없습니다: {images_dir}, {labels_dir}")

    ensure_dest_dirs(base_dir)
    if clear_dest:
        clear_dest_dirs(base_dir)

    images = list_images(images_dir, extensions)
    print("발견한 이미지 수:", len(images))
    if len(images) == 0:
        return (0, 0, 0)

    random.seed(seed)
    random.shuffle(images)

    split_idx = int(train_ratio * len(images))
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]
    print("분할 예정 - Train:", len(train_imgs), "장,", "Val:", len(val_imgs), "장")

    def process(img_list: List[str], split: str) -> int:
        total = len(img_list)
        processed = 0
        for i, img in enumerate(img_list, 1):
            label = img.rsplit(".", 1)[0] + ".txt"
            src_img = os.path.join(images_dir, img)
            src_lbl = os.path.join(labels_dir, label)

            if not os.path.exists(src_lbl):
                msg = f"라벨 누락: {label} (이미지: {img})"
                if skip_missing_labels:
                    print("[경고]", msg, "→ 건너뜀")
                    continue
                else:
                    raise FileNotFoundError(msg)

            dst_img = os.path.join(base_dir, "images", split, img)
            dst_lbl = os.path.join(base_dir, "labels", split, label)

            if dry_run:
                # 시뮬레이션 출력만
                if i % 50 == 0 or i == total:
                    print(f"[DRY-RUN {split}] {i}/{total} 예정")
                processed = i
                continue

            copy_or_move(src_img, dst_img, do_move)
            copy_or_move(src_lbl, dst_lbl, do_move)
            processed = i
            if i % 50 == 0 or i == total:
                print(f"[{split}] {processed}/{total} 처리")
        return processed

    train_count = process(train_imgs, "train")
    val_count = process(val_imgs, "val")
    total = train_count + val_count
    print(f"처리 완료 - Train: {train_count}장, Val: {val_count}장, 총 {total}장")
    print("Dataset prepared at:", base_dir)
    return (train_count, val_count, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split labeled YOLO dataset (images/labels) into train/val folders.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="layout_data/kexam",
        help="원본 내보내기 루트(하위에 images/, labels/ 존재)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="layout_data/kexam",
        help="출력 루트(하위에 images/{train,val}, labels/{train,val} 생성)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train 비율(0~1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="셔플 시드",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".jpg,.jpeg,.png",
        help="허용 이미지 확장자(콤마 구분)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="복사 대신 이동(shutil.move)",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="실행 전 대상(train/val) 폴더 비우기",
    )
    parser.add_argument(
        "--skip-missing-labels",
        action="store_true",
        help="라벨 누락 시 에러 대신 건너뜀",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 복사/이동 없이 계획만 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    split_dataset(
        export_dir=args.export_dir,
        base_dir=args.base_dir,
        train_ratio=args.train_ratio,
        extensions=exts,
        seed=args.seed,
        do_move=args.move,
        clear_dest=args.clear_dest,
        skip_missing_labels=args.skip_missing_labels,
        dry_run=args.dry_run,
    )