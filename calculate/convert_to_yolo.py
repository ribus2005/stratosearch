from pathlib import Path
import numpy as np
import cv2


BASE_DIR = Path("data")          # where .npy files live (same as reading.py)
OUT_DIR = Path("yolo_dataset")


def find_pairs(base: Path):
    """Ищет пары *_seismic.npy + *_labels.npy и возвращает список кортежей (key, seismic_path, labels_path)."""
    seismic = {}
    labels = {}

    for p in base.rglob("*.npy"):
        name = p.stem
        if name.endswith("_seismic"):
            key = name[:-len("_seismic")]
            seismic[key] = p
        elif name.endswith("_labels"):
            key = name[:-len("_labels")]
            labels[key] = p

    keys = sorted(seismic.keys() & labels.keys())
    return {k: (seismic[k], labels[k]) for k in keys}

def find_process_pairs(base: Path):
    raw_pairs = find_pairs(base)
    assert "test1" in raw_pairs.keys()
    assert "test2" in raw_pairs.keys()
    assert "train" in raw_pairs.keys()
    return {
        "train": raw_pairs["train"],
        "val": raw_pairs["test2"],
        "test": raw_pairs["test1"],
    }

def ensure_dirs(pairs: dict):
    keys = pairs.keys()
    dirs = [OUT_DIR / "images" / key for key in keys]
    dirs += [OUT_DIR / "labels" / key for key in keys]
    for p in dirs:
        p.mkdir(parents=True, exist_ok=True)

def laplacian(gray, blur_kernel: int = 0):
    res = cv2.Laplacian(gray, cv2.CV_64F)
    res = np.absolute(res)
    if blur_kernel > 0:
        res = cv2.blur(res, (blur_kernel, blur_kernel))
    return (res > 0).astype(np.uint8) * 255


def convert(pairs: dict):
    img_counter = 0
    for key, (seismic_path, labels_path) in pairs.items():
        print(f"\nProcessing pair: {key}")
        print(f"Labels path: {labels_path}")
        print(f"Seismic path: {seismic_path}")

        seismic = np.load(seismic_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")

        print("Read shapes:")
        print(f"Seicmic: {seismic.shape}")
        print(f"Labels: {labels.shape}")

        assert seismic.shape[0] == labels.shape[0] # Seismic / labels slice count mismatch
        n_slices = seismic.shape[0]
        print(f"Number of slices: {n_slices}")

        for slice_idx in range(n_slices):
            img = seismic[slice_idx]
            mask = labels[slice_idx]

            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_uint8 = img_norm.astype(np.uint8)

            img_name = f"{key}_{slice_idx:04d}.png"
            label_name = f"{key}_{slice_idx:04d}.png"
            img_path = OUT_DIR / "images" / key / img_name
            label_path = OUT_DIR / "labels" / key / label_name

            cv2.imwrite(img_path, img_uint8)
            # target = np.rot90(mask, -1)
            target = laplacian(mask, 2)
            cv2.imwrite(label_path, target)

            img_counter += 1

    print(f"\nDone. Total images written: {img_counter}")


if __name__ == "__main__":
    # convert()
    pairs = find_process_pairs(BASE_DIR)
    if not pairs:
        raise RuntimeError("No *_seismic.npy + *_labels.npy pairs found")
    # val_pairs = {"val": pairs["val"]}
    ensure_dirs(pairs)

    convert(pairs)
