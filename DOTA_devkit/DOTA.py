#The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
# import dota_utils as util
from . import dota_utils as util
from collections import defaultdict # 使用 defaultdict
import cv2
import math

def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DOTA:
    # 添加一个新的参数parseMode，决定解析标注坐标的格式
    def __init__(self, basepath, parseMode='parse_dota_poly'):
        self.basepath = basepath
        self.parseMode = parseMode # 决定了两点：1. 解析方法 2. showann坐标转换
        # 对应两种label，有向框与水平框
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)# 所有label的路径
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)# defaultdict(type): 
                                          # self.catToImgs为dict，对没有val的的key自动配一个空list
                                          # {'ship': ['P0706', 'P0706',...],...}
                                          # such like:
            
        self.ImgToAnns = defaultdict(list) 
        """
        such like :
        {
        'name': 'plane', 
        'difficult': '0', 
        'poly': [(643.0, 1048.0), (746.0, 1049.0), (748.0, 1186.0), (648.0, 1188.0)],
        'area': 14059.5
        }
        """
        self.createIndex(parseMode) # 初始化操作
    """
    createIndex方法：
    生成catToImgs字典，key为类别，val为imgid（list）
    生成catToAnns字典，key为imgid，val为包含的目标
    """
    def createIndex(self, parseMode):
        for filename in self.imgpaths:
            # 加一个检查是否为txt文件的语句
            filetype = filename.split('/')[-1].split('.')[-1]
            if filetype == 'txt':
                """
                通过label的路径解析每一个label.txt文件
                """
                # 全新的parse_dota将parseMode在util后台处理
                objects = util.parse_dota(filename,parseMode)
                imgid = util.custombasename(filename)
                self.ImgToAnns[imgid] = objects
                for obj in objects:
                    cat = obj['name']
                    self.catToImgs[cat].append(imgid)
            else:
                print(" Error: " + filename + " is not a .txt file")
                continue
        print(" DOTA dataset has been successfully loaded ")
                
    """ 
    getImgIds方法：
    输入需要查找的目标类别，返回所有包含指定类别（多目标要求包括所有种类）的图像的ID
    """
    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:  # catNums为空则返回所有img的id 
            print("加载图片ID完成：返回所有图片ID")
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms): # enumerate: 同时列出一个可迭代对象的数据和下标
                if i == 0: 
                    # 查找第一类时，将imgids初始化为所有包含此类目标的图片名set（图片名不重复）
                    imgids = set(self.catToImgs[cat])
                else:      
                    # 在其中查找包含后续类别的图片
                    # 意味着图片需要包含catNms指定的所有种类，才会被选中
                    # imgids &= set(self.catToImgs[cat])
                    # 将其改为求并集，只要包含其中一种目标，就选中
                    imgids |= set(self.catToImgs[cat])
        print("加载图片ID完成：共有 " + str(len(imgids)) + " 张图片符合筛选条件")
        return list(imgids)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects
    """
    重写showAnns方法，根据解析模式的不同，适配不同的显示模式
    """
    def showAnns(self,objects,imgId,range):
        if self.parseMode == 'parse_dota_rec':
            self.showRecAnns(objects,imgId,range)
        elif self.parseMode == 'parse_dota_rot_rect':
            self.showRbbAnns(objects,imgId,range)
        else:
            self.showNormAnns(objects, imgId, range)
    """
    showNormAnns方法：
    传入目标类型，图片id，
    """
    def showNormAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0] # imgId只加载一张图
        plt.imshow(img) # 画原图
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5 # 
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0] # 随机选颜色
            poly = obj['poly'] #目标框坐标
            polygons.append(Polygon(poly)) #append应该是允许多次画框
            color.append(c)
            point = poly[0] #记录起始点
            circle = Circle((point[0], point[1]), r) #以起始点为圆心画半径为r的圆（加粗点的效果）
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
    """
    loadImgs方法：
    传入imgids的list，返回图片序列(并不显示）
    """
    def loadImgs(self, imgids=[]): # 返回的是一个list，[img1,img2,...,imgn]，其中img为nparray
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        #print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        #print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.png')
            #print('filename:', filename)
            img = cv2.imread(filename,1) # png图像为RGBA四通道格式，参数取1则只加载RGB
            #print(img.shape)
            imgs.append(img)
        return imgs
    
    """
    showRbbAnns方法：
    显示采用[cx, cy, w, h, theta]表示的标注框
    """
    def showRbbAnns(self, objects, imgId, range):    
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        # pass
        
        img = self.loadImgs(imgId)[0] # imgId只加载一张图
        plt.imshow(img) # 画原图
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5 # 
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0] # 随机选颜色
            poly = obj['poly'] #目标框坐标
            poly = self.Rotationrec2xy(poly) # 从[cx, cy, w, h, theta]->[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            polygons.append(Polygon(poly)) #append应该是允许多次画框
            color.append(c)
            point = poly[0] #记录起始点
            circle = Circle((point[0], point[1]), r) #以起始点为圆心画半径为r的圆（加粗点的效果）
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        
    # Rotationrec2xy函数的中间件
    # 输入（cx,cy,w,h）,返回[[x1,x2,x3,x4],[y1,y2,y3,y4]]
    def rec2xy(self,rec):
        cx,cy,w,h = rec[0],rec[1],rec[2],rec[3]
        x = [cx-w/2,cx-w/2,cx+w/2,cx+w/2]
        y = [cy-h/2,cy+h/2,cy-h/2,cy+h/2]
        return x,y
    
    #仅用于画图（display）
    #输入[cx, cy, w, h, theta]
    #输出[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    def Rotationrec2xy(self, rec):
        cx,cy,w,h,theta = rec[0],rec[1],rec[2],rec[3],rec[4]
        RMatrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
        x_temp, y_temp = self.rec2xy([rec[0],rec[1],rec[2],rec[3]])
        box = np.array([x_temp,y_temp], dtype=np.float32)
        new_box = np.matmul(RMatrix,box-np.array([[cx],[cy]])) + np.array([[cx],[cy]])
        new_box = np.transpose(new_box)
        new_box = np.around(new_box) #取整，只用于显示，计算时还是保持float
        new_box[[2,3],:] = new_box[[3,2],:] #调整4个坐标点次序，保证画图正确
        return new_box.tolist()
    """
    showRecAnns():显示水平框
    """
    def showRecAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0] # imgId只加载一张图
        plt.imshow(img) # 画原图
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5 # 
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0] # 随机选颜色
            xmin, ymin, xmax, ymax = obj['bndbox'] # 转而解析['bndbox']
            poly =[(xmin,ymax),(xmin,ymin),(xmax,ymin),(xmax,ymax)]
            polygons.append(Polygon(poly)) #append应该是允许多次画框
            color.append(c)
            point = poly[0] #记录起始点
            circle = Circle((point[0], point[1]), r) #以起始点为圆心画半径为r的圆（加粗点的效果）
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
# if __name__ == '__main__':
#     examplesplit = DOTA('examplesplit')
#     imgids = examplesplit.getImgIds(catNms=['plane'])
#     img = examplesplit.loadImgs(imgids)
#     for imgid in imgids:
#         anns = examplesplit.loadAnns(imgId=imgid)
#         examplesplit.showAnns(anns, imgid, 2)