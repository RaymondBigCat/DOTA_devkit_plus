# TEST
"""
将DOTA标注(.txt)转化为XML格式
"""
import os
import scipy.misc as misc
from xml.dom.minidom import Document
import numpy as np
import copy, cv2
#%matplotlib inline
import matplotlib.pyplot as plt
import os
import pylab
import sys
# sys.path.append('/home/raymond/project/PytorchSSD_DOTA/data/DOTA_devkit') # 保证DOTA_devkit可用的关键
from DOTA_devkit import DOTA
import torchvision.transforms as transforms
from PIL import Image

def save_to_xml(save_path, file_name, img_shape, objects_axis, label_name, parseMode = 'parse_dota_rec'):
    """
    param:
        save_path: new path for saving xml file
        im_height: image height
        im_width: image width
        object_axis: 坐标点（4×2）
        lable_name：从int解码出str（类别）
        parseMode: 选择解析为旋转框or水平框
    return:
        (inplace)
    """
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(file_name)
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('raymond'))
    owner.appendChild(name_o)


    size = doc.createElement('size')
    annotation.appendChild(size)
    im_width,im_height,im_depth = img_shape
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        if parseMode == 'parse_dota_rec':
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(xmin)
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(ymin)
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(xmax)
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
            bndbox.appendChild(ymax)
        else:
            x0 = doc.createElement('x0')
            x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('y0')
            y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(y0)

            x1 = doc.createElement('x1')
            x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('y1')
            y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
            bndbox.appendChild(y1)
        
            x2 = doc.createElement('x2')
            x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
            bndbox.appendChild(x2)
            y2 = doc.createElement('y2')
            y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
            bndbox.appendChild(y2)

            x3 = doc.createElement('x3')
            x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
            bndbox.appendChild(x3)
            y3 = doc.createElement('y3')
            y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
            bndbox.appendChild(y3)
        
    f = open(save_path,'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close() 
    
def txt2xml(raw_folder, img_shape, class_list, parseMode='parse_dota_rec'):
    """
    Params:
        raw_folder: 原始数据根目录
    return：
        inplace
    """
    #建立文件夹
    save_folder = os.path.join(raw_folder, 'Annotations')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        print('Directory is built:'+ save_folder)
    dataset = DOTA.DOTA(raw_folder,parseMode)
    img_list = dataset.getImgIds(catNms=[])
    #txt_root = os.path.join(open_folder,'labelTxt')
    img_num = len(img_list)
    for index,img_id in enumerate(img_list): #迭代列表中所有文件
        #img = dataset.loadImgs(img_id)[0]
        labels = dataset.loadAnns(imgId=img_id)
        tmp_label = np.empty((0,5), int)
        for label in labels: #遍历当前文件中的所有目标框
            if label['name'] not in class_list:
                continue
            tmp = [int(x) for x in label['bndbox']]
            tmp.append(class_list.index(label['name'])) #通过list.index()方法找到类别的索引（class->int）
            tmp = np.array(tmp)
            tmp_label = np.vstack((tmp_label,tmp))
        save_path = os.path.join(save_folder,img_id+".xml")
        #img_shape = img.shape
        save_to_xml(save_path,img_id+'.png',img_shape, tmp_label, class_list, parseMode = 'parse_dota_rec')
        if index % (img_num/10) == 0:
            print(str(index)+' annotation files has finished!')

def generate_txt_imgids(origin_folder, dataset_name = 'train'):
    """
    xml_folder: .xml文件存放的路径
    dataset_name: 决定.txt的命名
    """
    txt_name = dataset_name + '.txt'
    txt_path = os.path.join(origin_folder,'ImageSets')
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    txt_path = os.path.join(txt_path,'Main')
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    txt_path = os.path.join(txt_path,txt_name)
    fp = open(txt_path,'w+')
    filelist = os.listdir(os.path.join(origin_folder,'images'))
    for xml in filelist:
        fp.write(os.path.splitext(xml)[0]+'\n')
    fp.close()