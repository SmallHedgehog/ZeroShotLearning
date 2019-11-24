import os.path as osp
import numpy as np
import torch
import torch.utils.data as Data

from PIL import Image

__all__ = ['CUB_200_2011']


class CUB_200_2011(Data.Dataset):
    def __init__(self, root_dir, phase='train', transform=None):
        super(CUB_200_2011, self).__init__()
        assert phase in ('train', 'val')

        self.root_dir = root_dir
        self.file = osp.join(root_dir, phase + '_' + 'classes.txt')
        with open(self.file) as file:
            self.interst_classes = [line.strip().split(' ')[0] for line in file.readlines()]
        self.classIdx2Idx = {class_idx: idx for idx, class_idx in enumerate(self.interst_classes)}
        self.Idx2classIdx = {idx: class_idx for idx, class_idx in enumerate(self.interst_classes)}

        self.transform = transform
        self.images_path = osp.join(root_dir, 'images')
        id_class_file = osp.join(root_dir, 'image_class_labels.txt')
        id_images_file = osp.join(root_dir, 'images.txt')
        attributes_file = osp.join(root_dir, 'attributes', 'class_attribute_labels_continuous.txt')

        id_images = self._read_file(id_images_file)

        self.set = list()
        with open(id_class_file) as file:
            for line in file.readlines():
                infos = line.strip().split(' ')
                if infos[1] in self.interst_classes:
                    self.set.append((id_images[infos[0]], self.classIdx2Idx[infos[1]]))

        # Normalize attributes
        source_norm_attris, target_norm_attris = self._normalize_attris(attributes_file, True)
        if phase == 'train':
            self.attributes = torch.FloatTensor(source_norm_attris)
        else:
            self.attributes = torch.FloatTensor(target_norm_attris)

    def _read_file(self, read_file):
        dct = dict()
        with open(read_file) as file:
            for line in file.readlines():
                infos = line.strip().split(' ')
                dct[infos[0]] = infos[1]
        return dct

    def _normalize_attris(self, attributes_file, mean_correction=True):
        source_file = osp.join(self.root_dir, 'train_classes.txt')
        target_file = osp.join(self.root_dir, 'val_classes.txt')
        source_idx = [(int(line.strip().split(' ')[0]) - 1) for line in open(source_file)]
        target_idx = [(int(line.strip().split(' ')[0]) - 1) for line in open(target_file)]

        codes = np.loadtxt(attributes_file).astype(float)
        if codes.max() > 1:
            codes /= 100.
        code_mean = codes[source_idx, :].mean(axis=0)
        for s in range(codes.shape[1]):
            codes[codes[:, s] < 0, s] = code_mean[s] if mean_correction else 0.5
        # Mean correction
        if mean_correction:
            for s in range(codes.shape[1]):
                codes[:, s] = codes[:, s] - code_mean[s] + 0.5
        return codes[source_idx], codes[target_idx]

    @property
    def get_class_attributes(self):
        return self.attributes

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        image_file = osp.join(self.images_path, self.set[index][0])
        image_label = int(self.set[index][1])

        image = Image.open(image_file).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, image_label
