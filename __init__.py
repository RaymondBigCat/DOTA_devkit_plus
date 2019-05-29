# from .voc import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
import sys
from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .coco import COCODetection
from .data_augment import *
from .config import *
# sys.path.append('/home/raymond/project/PytorchSSD_DOTA/data/DOTA_devkit') # 保证DOTA_devkit可用的关键
#from .DOTA_devkit import dota_utils as util
#from DOTA_devkit import DOTA
#sys.path.append('/home/raymond/project/PytorchSSD_DOTA/data/DOTA_devkit')
from .dota import DOTADetection, DotaAnnTrans
