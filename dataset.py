# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import json

import torch.utils.data.dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils import ImageTransforms, convert_image

class SRDataset(Dataset):
    """
    Кастомный датасет, который будет использоваться в DataLoader PyTorch-а.
    """

    def __init__(self, crop_size, scaling_factor, lr_img_type, hr_img_type, 
                 augments={'rotation': False, 'hflip': False}, train_data_name=None):
        self.augments = augments
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.train_data_name = train_data_name

        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        with open(self.train_data_name, 'r') as j:
            self.images = json.load(j)

        # Select the correct set of transforms
        self.transform = ImageTransforms(crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type,
                                         augments=self.augments)

    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.images)


class DatasetFromFolder(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, input_dir, target_dir):
        r"""

        Args:
            input_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
        """
        super(DatasetFromFolder, self).__init__()
        self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        input = self.transforms(Image.open(self.input_filenames[index]))
        target = self.transforms(Image.open(self.target_filenames[index]))

        return input, target

    def __len__(self):
        return len(self.input_filenames)


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.

    """
    return any(filename.endswith(extension) for extension in ["bmp", ".png",
                                                              ".jpg", ".jpeg",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])
