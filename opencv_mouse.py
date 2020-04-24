import cv2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

drawing = False
# 如果mode为true绘制矩形。按下'm' 变成绘制曲线。
ix, iy = -1, -1

# 创建回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy,drawing

    # 当按下左键是返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True


    # 当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        # 当鼠标松开停止绘画。
        elif event == cv2.EVENT_LBUTTONUP:
            drawing == False
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)



# 回调函数与OpenCV 窗口绑定在一起,
# 在主循环中我们需要将键盘上的“m”键与模式转换绑定在一起。
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
# 绑定事件
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break