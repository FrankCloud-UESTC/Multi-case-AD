import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import torch

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width, class_name):
    image_decoded = cv2.imread(filename)
    label_dir = filename.replace(class_name, "labels").split('.')[0] + ".png"
    if os.path.exists(label_dir):
        print(label_dir)
        label = cv2.imread(label_dir)
        label = label[:, :, :1]
        label = label / np.max(label)
        print("OK to labels")
    else:
        label = np.zeros([image_decoded.shape[0], image_decoded.shape[1], 1])

    # 图像的resize
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
    image_resized = np.moveaxis(image_resized, 2, 0)

    # label的resize 
    size = [8, 4]
    labels = []
    for s in size:
        label_ = cv2.resize(label, (resize_width // s, resize_height // s))
        label_ = label_.astype(dtype=np.float32)
        if len(label_.shape) == 2:
            label_ = np.expand_dims(label_, 2)
        label_ = np.moveaxis(label_, 2, 0)
        label_[label_ > 0] = 1
        label_[label_ <= 0] = 0
        labels.append(torch.tensor(label_))
    img_name = filename.split('/')[-1]
    return image_resized, labels, img_name


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, class_name=""):
        self.dir = video_folder
        self.transform = None
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._className = class_name
        self.setup()
        self.samples = self.get_all_samples()
        print("trained on " + str(len(self.samples)) + " samples")

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.png'))
            self.videos[video_name]['frame'].sort()

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])):
                frames.append(self.videos[video_name]['frame'][i])
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        batch = []
        for i in range(1):
            image, label, img_name = np_load_frame(self.videos[video_name]['frame'][index], self._resize_height,
                                                   self._resize_width, self._className)
            if self.transform is not None:
                batch.append(self.transform(image))
            else:
                batch.append(torch.tensor(image))
            return np.concatenate(batch, axis=0), label, img_name

    def __len__(self):
        return len(self.samples)
