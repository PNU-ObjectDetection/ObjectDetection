from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.image_ids = []
        folder_path = self.root_dir + '/' + self.set_name
        os.chdir(folder_path)
        files = os.listdir(folder_path)

        for f in files:  # 3129
          if len(f) == 22: # .json
            self.coco = COCO(os.path.join(folder_path, f))

        self.image_ids = self.coco.getImgIds()
        if self.set_name == 'test' or self.set_name == 'val':
          self.image_ids = self.image_ids[-100:]
 
        self.load_classes()
        os.chdir('/content/drive/My Drive/EfficientDet.Pytorch')

    def load_classes(self):
        # load class names (name -> label)
        # categories = self.coco.loadCats(self.coco.getCatIds())
        # categories.sort(key=lambda x: x['id'])
        categories = [{'id':0, 'name' : '운동화'}]
    
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['identifier'])

        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations
        
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        coco_annotations = coco_annotations[0]
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['boxcorners'][2] < 1 or a['boxcorners'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['boxcorners']
            # annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotation[0, 4] = self.coco_label_to_label(0)
            annotations = np.append(annotations, annotation, axis=0)

        # # transform from [x, y, w, h] to [x1, y1, x2, y2]
        # annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        # annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        # return self.coco_labels[label]
        return self.coco_labels[0]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 1


if __name__ == '__main__':
    from augmentation import get_augumentation
    dataset = CocoDataset(root_dir='/root/data/coco', set_name='trainval35k',
                          transform=get_augumentation(phase='train'))
    sample = dataset[0]
    print('sample: ', sample)
