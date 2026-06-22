from torch.utils.data import DataLoader
from ImageDataset import  *
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
import logging

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2():
    return Compose([
        _convert_image_to_rgb,
        # AdaptiveResize(224), #512
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3():
    return Compose([
        _convert_image_to_rgb,
        # AdaptiveResize(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger

def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)

def openreadtxt(file_name):
    data = []
    ref_name = []
    dist_name = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(',')  # 按‘，’切分每行的数据
        ref_name.append(tmp_list[0])
        dist_name.append(tmp_list[1])
        data.append(float(tmp_list[2]))  # 将每行数据插入data中
    return ref_name,dist_name,data

def set_dataset_pyiqa(dataset, csv_file, bs, data_set, num_workers, preprocess, mos_type, test, blind=False):
    if dataset == 'AGIQA-1k':
        data = AGIQA1kDataset_pyiqa(
            csv_file=csv_file,
            img_dir=data_set,
            test=test,
            preprocess=preprocess,
            blind=blind)

    elif dataset == 'AGIQA-3k':
        data = AGIQA3kDataset_pyiqa(
            csv_file=csv_file,
            img_dir=data_set,
            test=test,
            preprocess=preprocess,
            mos_type = mos_type,
            blind=blind)

    elif  dataset == 'AIGCIQA2023':
        data = AIGCIQA2023Dataset_pyiqa(
            csv_file=csv_file,
            img_dir=data_set,
            test=test,
            preprocess=preprocess,
            mos_type=mos_type,
            blind=blind)

    elif  dataset == 'PKU-AIGIQA':
        data = PKUAIGIQADataset_pyiqa(
            csv_file=csv_file,
            img_dir=data_set,
            test=test,
            preprocess=preprocess,
            mos_type=mos_type,
            blind=blind)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader