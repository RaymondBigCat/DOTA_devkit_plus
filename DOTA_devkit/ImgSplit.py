import os
import codecs
import numpy as np
import math
# from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo #shapely：地理空间几何库
# import dota_utils as util
from . import dota_utils as util
import copy

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=100,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.png',
                 parseMode = 'parse_dota_poly2'
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.parseMode = parseMode
        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou
    '''
    half_iou：交集/poly1_area（面积比）
    inter_poly = 交集（点集）
    '''
    
    '''
    saveimagepatches:保存子图
    '''
    def saveimagepatches(self, img, subimgname, left, up):
        # 深拷贝子图
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    # 子图与子标注的保存操作
    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        '''
        param resizeimg:
        param objects:
        param subimgname:
        param left,up,right,down:
        '''
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        # 子图
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                # 标注框（x,y,x,y,x,y,x,y）
                # 旋转框需要将格式转化
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                # 快捷的求面积method
                if (gtpoly.area <= 0):
                    continue
                #计算IoU ？ self.calchalf_iou ?
                # inter_poly, half_iou是什么？
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                
                # half_iou == 1:该目标被完全包含在子图中
                if (half_iou == 1):
                    # self.polyorig2sub ? 
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                
                # 该目标被部分包含在子图中（looks like 本来是有阈值限制的，现在被取消了，只要有一点包含就算） 
                elif (half_iou > self.thresh):
                #elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    # 计算目标框剩下的边界点
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota(fullname,self.parseMode)
        # 1. 先将label全部resize
        for obj in objects:
            # lambda匿名函数标识符，相当于临时定义了一个函数（因为有些函数的参数需要传入一个函数，所以有用）
            # map/reduce：传入两个参数，将函数应用于对象上
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))
        # 2. 再将img resize    
        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        row = col = 0 # row：行，col：列
        # loop1：从左往右扫（列）
        while (left < weight):
            # subsize > 原始框的宽（最后一列）
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            row = 0 # 重置子图行数（Y轴）
            # loop2：从上往下扫（行）
            while (up < height):
                # 最后一行
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                # 子图像命名
                # subimgname = outbasename + str(left) + '___' + str(up)
                subimgname = outbasename + str(col) + '__' + str(row)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                # 保存子图，用到savepatches
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                # 
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
                row += 1 # 更新子图行数（Y轴）
            col += 1 # 更新子图列数（X轴）
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
                
    # 缩放再切割
    def splitdata(self, rate, imglist=None):
        """
        :param rate: resize rate before cut
        """
        if imglist == None:
            imagelist = util.GetFileFromThisRootDir(self.imagepath)
            imglist = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        for name in imglist:
            self.SplitSingle(name, rate, self.ext)

if __name__ == '__main__':
    # example usage of ImgSplit
    split = splitbase(r'example',
                       r'examplesplit')
    split.splitdata(1)