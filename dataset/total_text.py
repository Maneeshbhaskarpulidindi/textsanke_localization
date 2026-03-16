import scipy.io as io
import numpy as np
import os

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        # self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        self.annotation_list = [f"poly_{os.path.splitext(img_name)[0]}.mat" for img_name in self.image_list ## changed from poly_gt to poly subhra##
]

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            
            # Correctly extract text as a Python string
            text_raw = cell[4][0]
            if len(text_raw) > 0:
                # If text_raw is a numpy array (e.g., array(['TEXT'])), extract the string
                if isinstance(text_raw, np.ndarray):
                    text = str(text_raw[0])
                else: # It might already be a string
                    text = str(text_raw)
            else:
                text = '#' # Mark as ignored if text field is empty
            
            # print(f"DEBUG: Extracted text: '{text}', Type: {type(text)}") # Debug print

            ori_raw = cell[5][0]
            if len(ori_raw) > 0:
                if isinstance(ori_raw, np.ndarray):
                    ori = str(ori_raw[0])
                else:
                    ori = str(ori_raw)
            else:
                ori = 'c'

            if len(x) < 4:  # too few points
                continue
            
            # Filter out polygons with '###' or '#' text (ignored regions)
            if text == '###' or text == '#':
                continue

            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='data/total-text',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, instance_label_map, meta = trainset[idx]
        print(idx, img.shape)