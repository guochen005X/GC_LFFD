import cv2
import csv
import os
import os.path as osp
face_casecade = cv2.CascadeClassifier('E:\\deep_learn\\GC_LFFD\\haarcascade_frontalface_default.xml')
def StaticDetect(filename):
    '''
    静态图像的人脸检测
    '''
    # 创建一个级联分类器，加载一个 .xml文件，它既可以是Haar特征，也可以是LBP特征的分类器
    #face_casecade = cv2.CascadeClassifier('E:\\deep_learn\\GC_LFFD\\haarcascade_frontalface_default.xml')
    global face_casecade

    # 加载图像

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    detectMultiScale进行人脸检测
    传入参数为args：
                    img:传入图像
                    object：被检测的物体的矩形框向量组
                    scaleFactor：表示前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1，即每次搜索窗口扩大10%
                    minNegihbors，表示构成检测目标的相邻矩形的最小个数(默认为3个)
                    flags:要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为CV_HAAR_DO_CANNY_PRUNING，那么函数会使用Canny边缘检测来排除边缘过多或者过少的区域，这些通常不会是人脸所在区域
                    minSize和maxSize：用来限制得到的目标区域的范围
    输出为：vector保存各个人脸的坐标、大小（用矩形表示）
    '''
    faces = face_casecade.detectMultiScale(gray_img, 1.2, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.namedWindow('Face_Detected')
        # cv2.imshow('Face_Detected', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return True, x,y,w,h
    else:
        #cv2.destroyAllWindows()
        return False,0,0,0,0




def getAllImgPath(dirName,csvName):
    assert osp.exists(dirName) , 'dir_name Is Empty !'
    currunt_index = 0
    with open(csvName, 'w', newline='') as cvsfile:
        files = csv.writer(cvsfile)
        filenames = os.listdir(dirName)
        for filename in filenames:
            imgNames =  os.listdir(osp.join(dirName, filename))
            for imgName in imgNames:
                currunt_index += 1
                print('Currunt Image Index : ')
                print(currunt_index)
                fullImgPath = osp.join(dirName, filename, imgName)
                result ,x,y,w,h = StaticDetect(fullImgPath)
                if result:
                    files.writerow([fullImgPath, x,y,w,h])
        cvsfile.close()


csv_name = 'E:\\deep_learn\\GC_LFFD\\Detect_Face.csv'
dirName = "G:\\DATESETS\\64_CASIA-FaceV5\\data"
getAllImgPath(dirName,csv_name)
# filename = 'E:\\deep_learn\\GC_LFFD\\000_0.bmp'
# StaticDetect(filename)