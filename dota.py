import pickle
import sys

import cv2
import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
sys.path.append('/home/raymond/project/PytorchSSD_DOTA/data') # 保证DOTA_devkit可用的关键
import torchvision.transforms as transforms
from PIL import Image
#from DOTA_devkit import dota_utils as util
from DOTA_devkit import DOTA
# from .voc_eval import voc_eval # VOCdevkit
from config import DOTA_CLASSES
"""
VOC_CLASSES = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
"""
"""
DOTA_CLASSES = ('__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
"""
# DOTA_CLASSES = ('__background__', 'plane')
class DotaAnnTrans:
    """Transforms a DOTA annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True,parseMode = 'parse_dota_rec'):
        self.class_to_ind = class_to_ind or dict(
            zip(DOTA_CLASSES, range(len(DOTA_CLASSES))))
        self.keep_difficult = keep_difficult
        self.parseMode = parseMode
        if self.parseMode == 'parse_dota_rec':
            self.parsekw = 'bndbox'
        else:
            self.parsekw = 'poly'
    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an DOTA.anns
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        '''
        在此详细解析label文件
        
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            # 坐标框append
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
        '''
        labels = []
        # 解析DOTA.loadAnns返回的Anns
        if self.parseMode == 'parse_dota_rec':
            for num,ann in enumerate(target):
                labels.append([])
                labels[num].extend(list(ann['bndbox']))
                labels[num].append(self.class_to_ind[ann['name']])
        # 通过np的vstack进行垂直方向的数组叠加
        if labels != []: #关键。负样本的标注为0,0,0,0,0
            res = np.vstack((res, labels))  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
        #
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ] 

class DOTADetection(data.Dataset):
    """DOTA Detection Dataset Object

    input is image, target is annotation

    Arguments:
        rootPath (string): filepath to DOTA dataset folder, will be '/media/raymond/MainDrive/Dataset/DOTA'
        
        image_set (string): imageset to use (eg. 'train', 'val', 'test', 'train_test')
            (default: 'train')
            
        (None) transform (callable, optional): transformation to perform on the
            input image
            
        preproc : pre-procced of images(eg: data augment) and annotations
            (default: None) （在train.py中被调用，preproc类在data_augment.py里）
            
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
            
        dataset_name (string, optional): which dataset to load
            (default: 'DOTA')
            
        parseMode: choose the format for parsing the annotation
            (default:'parse_dota_rec', which anns will be parsed as [xmin, ymin, xmax, ymax] 
            
        catNms: choose the category of obejcts that will be loaded
            (default: [] , means all)
    """

    def __init__(self, rootPath, image_sets='train', preproc=None, target_transform=None,
                 dataset_name='DOTA', parseMode='parse_dota_rec', catNms=list(DOTA_CLASSES)):
        self.rootPath = rootPath
        self.image_set = image_sets
        # 构造DOTA加载路径
        self.path = os.path.join(self.rootPath, self.image_set)
        # 预处理，默认None
        self.preproc = preproc
        # target_transform = AnnotationTransform
        self.target_transform = target_transform
        self.name = dataset_name
        # 解析方式
        self.parseMode = parseMode
        # DOTA筛选类别
        self.catNms = catNms
        print('loading classes :')
        for item in range(1,len(catNms)):
            print(catNms[item])
        # 加载DOTA(imgIDs, anns)
        self.dataset = DOTA.DOTA(self.path, parseMode=self.parseMode)
        self.imgIDs = self.dataset.getImgIds(catNms=self.catNms) #取所有图，无用标注图作为负样本
        # 将类别编码为数字
        # self.class_to_ind = dict(zip(DOTA_CLASSES,range(len(DOTA_CLASSES))))
        
    def __getitem__(self, index):
        img_id = self.imgIDs[index]
        # print('loading image:%s'%img_id)
        # 调用DOTA devkit 的方法load imgs（背后是cv2的imread）
        img = self.dataset.loadImgs(img_id)[0]
        target = self.dataset.loadAnns(imgId=img_id,catNms=self.catNms) #标注只取有用的
        # target 即是 Label
        # 
        # img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        # height, width, _ = img.shape
        
        '''
        详细解析Anns'''
        if self.target_transform is not None:
            target = self.target_transform(target)  
        '''
        数据增强，resize图像等一系列操作都在data_augment的preproc里面'''
        if self.preproc is not None:
            # preproc 
            img, target = self.preproc(img, target)
            # print(img.size())

            # target = self.target_transform(target, width, height)
        # print(target.shape)

        return img, target

    def __len__(self):
        return len(self.imgIDs)
    
    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.imgIDs[index]
        img = self.dataset.loadImgs(img_id)[0]
        return img
    
    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.imgIDs[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt