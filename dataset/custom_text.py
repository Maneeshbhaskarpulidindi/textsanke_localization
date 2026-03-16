"""
CustomTextDataset — loads PNG images + per-image .txt polygon labels.

Label format (one polygon per line):
    x1,y1,x2,y2,...,xN,yN,TEXT

The last comma-separated token is the text transcription; all preceding
tokens are alternating x,y integer coordinates.
"""

import os
import numpy as np

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance


class CustomTextDataset(TextDataset):
    """
    Dataset that reads custom .txt polygon annotations paired with PNG images.

    Directory layout expected:
        data_root/
            images/   *.png
            labels/   *.txt  (same stem as image)

    Args:
        data_root  (str): path to the root folder (containing images/ and labels/)
        is_training (bool): if True use 80% for training, else 20% for validation
        transform: augmentation / base transform callable
        val_split  (float): fraction reserved for validation (default 0.2)
    """

    def __init__(self, data_root, is_training=True, transform=None, val_split=0.2):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        self.image_dir = os.path.join(data_root, "images")
        self.label_dir = os.path.join(data_root, "labels")

        # Collect all images that have a matching label file
        all_images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(".png")
        ])

        valid_pairs = []
        for img_name in all_images:
            stem = os.path.splitext(img_name)[0]
            lbl_name = stem + ".txt"
            lbl_path = os.path.join(self.label_dir, lbl_name)
            if os.path.isfile(lbl_path):
                valid_pairs.append((img_name, lbl_name))

        # Deterministic train/val split
        split_idx = int(len(valid_pairs) * (1.0 - val_split))
        if is_training:
            self.pairs = valid_pairs[:split_idx]
        else:
            self.pairs = valid_pairs[split_idx:]

        print(f"[CustomTextDataset] {'Train' if is_training else 'Val'} set: "
              f"{len(self.pairs)} samples (total paired: {len(valid_pairs)})")

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def parse_txt(self, label_path):
        """
        Parse a single .txt annotation file.

        Each line format:
            x1,y1,x2,y2,...,xN,yN,TEXT

        Returns:
            list of TextInstance
        """
        polygons = []
        with open(label_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = line.split(",")
                # Last token is text, everything before are numeric coords
                text = tokens[-1]
                coord_tokens = tokens[:-1]

                # Must have an even number of coordinate values
                if len(coord_tokens) < 8 or len(coord_tokens) % 2 != 0:
                    continue  # need at least 4 points (8 values)

                try:
                    coords = [int(v) for v in coord_tokens]
                except ValueError:
                    continue  # skip malformed lines

                pts = np.array(coords, dtype=np.int32).reshape(-1, 2)  # (N, 2)

                if pts.shape[0] < 4:
                    continue

                polygons.append(TextInstance(pts, "c", text))

        return polygons

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        img_name, lbl_name = self.pairs[idx]

        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, lbl_name)

        image = pil_load_img(image_path)
        polygons = self.parse_txt(label_path)

        return self.get_training_data(
            image, polygons,
            image_id=img_name,
            image_path=image_path
        )

    def __len__(self):
        return len(self.pairs)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from util.augmentation import Augmentation

    data_root = sys.argv[1] if len(sys.argv) > 1 else "dummy 1/dummy"

    ds = CustomTextDataset(
        data_root=data_root,
        is_training=True,
        transform=Augmentation(
            size=512,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    )
    print(f"Dataset length: {len(ds)}")
    img, tm, tr, tcl, rm, sm, cm, inst_map, meta = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Image id    : {meta['image_id']}")
    print("Smoke-test PASSED.")
