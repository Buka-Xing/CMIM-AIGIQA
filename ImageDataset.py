import os
import torch
import functools
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def image_loader(image_name):
    # print(image_name)
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')

def get_default_img_loader():
    return functools.partial(image_loader)

class AGIQA1kDataset_pyiqa(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 test,
                 blind=False,
                 get_loader=get_default_img_loader):

        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test
        # self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        I = F.interpolate(I, size=224, mode='bilinear', align_corners=False)

        mos = float(self.data.iloc[index, 2])
        prompt = self.data.iloc[index, 1]

        sample = {'img':I, 'prompt': prompt, 'mos': mos}
        return sample

    def __len__(self):
        return len(self.data.index)

class AGIQA3kDataset_pyiqa(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 test,
                 mos_type,
                 blind=False,
                 get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test
        self.mos_type = mos_type
        # self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        I = F.interpolate(I, size=224, mode='bilinear', align_corners=False)

        if self.mos_type == 'quality':
            mos = float(self.data.iloc[index, 5])  # MOS_quality
        elif self.mos_type == 'consis':
            mos = float(self.data.iloc[index, 7])  # MOS_consis
        prompt = self.data.iloc[index, 1]

        sample = {'img':I, 'prompt': prompt, 'mos': mos}
        return sample

    def __len__(self):
        return len(self.data.index)

class AIGCIQA2023Dataset_pyiqa(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 test,
                 mos_type,
                 blind=False,
                 get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test
        self.mos_type = mos_type
        # self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 0] + '/' + self.data.iloc[index, 1]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        I = F.interpolate(I, size=224, mode='bilinear', align_corners=False)

        if self.mos_type == 'quality':
            mos = float(self.data.iloc[index, 2])  # MOS quality
        elif self.mos_type == 'authen':
            mos = float(self.data.iloc[index, 3])  # MOS authentic
        elif self.mos_type == 'consis':
            mos = float(self.data.iloc[index, 4])  # MOS consistency
        prompt = self.data.iloc[index, 5]

        sample = {'img':I, 'prompt': prompt, 'mos': mos}
        return sample

    def __len__(self):
        return len(self.data.index)

class PKUAIGIQADataset_pyiqa(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 test,
                 mos_type,
                 blind=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, sep=',', header=None)
        self.data = self.data.iloc[1:]
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test
        self.mos_type = mos_type
        self.blind = blind
        self.in_memory = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = self.data.iloc[index, 2]
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        I = F.interpolate(I, size=224, mode='bilinear', align_corners=False)

        if self.mos_type == 'quality':
            mos = float(self.data.iloc[index, 3])  # quality
        elif self.mos_type == 'authen':
            mos  = float(self.data.iloc[index, 4])  # authenticity
        elif self.mos_type == 'consis':
            mos  = float(self.data.iloc[index, 5])  # consistenty
        prompt = self.data.iloc[index, 1]

        sample = {'img':I, 'prompt': prompt, 'mos': mos}
        return sample

    def __len__(self):
        return len(self.data.index)