import cv2
import numpy as np

"""###############################图像预处理###############################################"""
# 反相灰度图，将黑白阈值颠倒,挨个像素处理
def accessPiexl(img):
    # change_gray(img)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img

# 反相二值化图像

def accessBinary(img):
    img = accessPiexl(img)#反色

    kernel = np.ones((3, 3), np.uint8)
    #滤波
    # img = cv2.medianBlur(img, 3)#均值滤波 即当对一个值进行滤波时，使用当前值与周围8个值之和，取平均做为当前值
    img = cv2.GaussianBlur(img, (3, 3), 0)#高斯滤波 根据高斯的距离对周围的点进行加权,求平均值1，0.8， 0.6， 0.8
    # img = cv2.medianBlur(img, 3)#中值滤波 将9个数据从小到大排列，取中间值作为当前值

    # 进行腐蚀操作，去除边缘毛躁
    # img = cv2.erode(img, kernel, iterations=1)

    #利用阈值函数，二值化
    # _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)#简单二值化，阈值固定
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -9)#自适应滤波   这个效果还行

    # 边缘膨胀，不加也可以
    img = cv2.dilate(img, kernel, iterations=1)#被执行的次数
    return img




# 显示结果及边框
def showResults(path, borders, re_path,results=None):
    img = cv2.imread(path)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 画出矩形，各参数依次是：img是原图，第二个参数是矩阵的左上点坐标，
        # 第三个参数是矩阵的右下点坐标，第四个参数是画线对应的rgb颜色，第五个参数是所画的线的宽度。
        # print(type(results[0]))
        if results != None:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
            # img – 想要打印上文字的图像  text – 想要打印的文字  org – 文字的左下角坐标
            # fontFace – 字体，可选的有：FONT_HERSHEY_SIMPLEX,
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.imwrite(re_path, img)
#####################################################################


# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(path, maxArea=100):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    #这里关于返回值是几个网上不太确定，但是本工程是2个
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            if h > 20:
                border = [(x, y), (x + w, y + h)]
                borders.append(border)
    return borders
 
def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        h = abs(border[0][1] - border[1][1])#高度
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        h_extend = h // 5
        # print(h_extend)
        targetImg = cv2.copyMakeBorder(borderImg, h_extend, h_extend, int(extendPiexl*1.1), int(extendPiexl*1.1), cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData

