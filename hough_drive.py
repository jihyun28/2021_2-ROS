#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################################################
# 프로그램명 : hough_drive_c1.py
# 작 성 자 : (주)자이트론
# 생 성 일 : 2020년 07월 23일
# 본 프로그램은 상업 라이센스에 의해 제공되므로 무단 배포 및 상업적 이용을 금합니다.
####################################################################

import rospy, rospkg, time
import numpy as np
import cv2, math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os

def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

image = np.empty(shape=[0])
bridge = CvBridge()
motor = None
Width = 320
Height = 240
Offset = 160#관심영역 높이, 너비 알아서
Gap = 40

cnt = 0 #우승팀코드 참고해서 추가 stopline lab_count

starting = time.time() #우승팀 시작할 때 현재시간
ending = 0 #우승팀 한바퀴라고 예상했을 때의 현재 시간

interval = 0 #우승팀 한바퀴 도는데 얼마나 걸렸는지

cam = False
cam_debug = True

sub_f = 0
time_c = 0

def img_callback(data):
    global image   
    global sub_f 
    global time_c

    sub_f += 1
    if time.time() - time_c > 1:
        #print("pub fps :", sub_f)
        time_c = time.time()
        sub_f = 0

    image = bridge.imgmsg_to_cv2(data, "bgr8")

# publish xycar_motor msg
def drive(Angle, Speed): 
    global motor

    motor_msg = xycar_motor()
    motor_msg.angle = Angle
    motor_msg.speed = Speed

    motor.publish(motor_msg)
    template()

# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), (0, 255, 0), 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 2, 7 + offset),
                       (lpos + 2, 12 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 2, 7 + offset),
                       (rpos + 2, 12 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-2, 7 + offset),
                       (center+2, 12 + offset),
                       (0, 255, 0), 2)    
    cv2.rectangle(img, (157, 7 + offset),
                       (162, 12 + offset),
                       (0, 0, 255), 2)
    return img

# left lines, right lines
def divide_left_right(lines):
    global Width

    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)
        
        if low_slope_threshold < abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []
    th = 25

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width/2 - th):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + th):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines, slope

# get average m, b of line, sum of x, y, mget lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap, cam_debug

    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    
    m = 0
    b = 0

    if size != 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += float(y2 - y1) / float(x2 - x1)

        x_avg = x_sum / (size * 2)
        y_avg = y_sum / (size * 2)

        m = m_sum / size
        b = y_avg - m * x_avg

    if m == 0 and b == 0:
        if left:
            pos = 0
        elif right:
            pos = Width
    else:
        y = Gap / 2

        pos = (y - b) / m

        if cam_debug:
            b += Offset
            xs = (Height - b) / float(m)
            xe = ((Height/2) - b) / float(m)

            cv2.line(img, (int(xs), Height), (int(xe), (Height/2)), (255, 0,0), 3)

    return img, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    global cam, cam_debug, img
    global starting, ending
    global cnt, interval

    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    roi = gray[Offset : Offset+Gap, 15 : Width-15] #roi = gray[Offset : Offset+Gap, 0 : Width]

    # blur
    kernel_size = 3
    standard_deviation_x = 1.5     #Kernel standard deviation along X-axis
    blur_gray = cv2.GaussianBlur(roi, (kernel_size, kernel_size), standard_deviation_x)

    # canny edge
    low_threshold = 70
    high_threshold = 150
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold, kernel_size)

    # HoughLinesP #직선이라고 생각되는 선 다 그음 OpenCV
    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,30,30,10)

    if cam:
        cv2.imshow('calibration', frame)
    # divide left, right lines
    if all_lines is None:
        return (Width)/2, (Width)/2, False
    left_lines, right_lines, slope = divide_left_right(all_lines)
    print("slope", slope)
    ending = time.time()
    interval = (ending - starting)#처음 가동한 시간과 현재 시간의 차이
    print("interval : ", interval)
    #내가 쓴 거
    #if abs(slope) < 0.3: #코너에서 너무 꺾였을 때
     # print "slope<0.3"
      #stopline detected"
     
    #if interval>8.0:
     # cnt+=1
      #print("cnt", cnt)
    
    # get center of lines
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    if cam_debug:
        # draw lines
        frame = draw_lines(frame, left_lines)
        frame = draw_lines(frame, right_lines)
        frame = cv2.line(frame, (115, 117), (205, 117), (0,255,255), 2)

        # draw rectangle
        frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
        frame = cv2.rectangle(frame, (0, Offset), (Width, Offset+Gap), (0, 255, 0), 2)

    img = frame
    if interval >= 37.7:
      print("car has stop")
      return lpos, rpos, False
    #elif abs(slope) >=1.0:
     # return lpos, rpos, 2     

    return lpos, rpos, True


def draw_steer(steer_angle):
    global Width, Height, img
    #img = cv_image

    arrow = cv2.imread('/home/pi/xycar_ws/src/auto_drive/src/steer_arrow.png')

    origin_Height = arrow.shape[0]
    origin_Width = arrow.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height/2
    arrow_Width = (arrow_Height * 462)/728

    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (-steer_angle) * 1.5, 0.7)    
    arrow = cv2.warpAffine(arrow, matrix, (origin_Width+60, origin_Height))
    arrow = cv2.resize(arrow, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = img[arrow_Height: Height, (Width/2 - arrow_Width/2) : (Width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow)
    img[(Height - arrow_Height): Height, (Width/2 - arrow_Width/2): (Width/2 + arrow_Width/2)] = res

    cv2.imshow('steer', img)

# template matching
def template():
  # img_t = cv2.imread('../img/figures.jpg') #edge_img
  img_t = edge_img
  template = cv2.imread('../img/taekwonv1.jpg') #캡처이미지
  th, tw = template.shape[:2]
  cv2.imshow('template', template)

  methods = ['cv2.TM_CCOEFF_NORMED']
  for i, method_name in enumerate(methods):
    img_draw = img_t.copy()
    method = eval(method_name)
    # 템플릿 매칭   ---①
    res = cv2.matchTemplate(img_t, template, method)
    # 최대, 최소값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    # 매칭 좌표 구해서 사각형 표시   ---④      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)
    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left, \
                cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)

def start():
    global motor
    global image
    global Width, Height

    rospy.init_node('auto_drive')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    #motorS = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)
    #image_subS = rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
    print "---------- Xycar C1 HD v1.0 ----------"
    time.sleep(3)

    #sq = rospy.Rate(30)

    t_check = time.time()
    f_n = 0

    while not rospy.is_shutdown():
        print "---------- while not rospy.is_shutdown----------"
        while not image.size == (Width*Height*3):
            print "---------- while not image.size ----------"
            continue

        f_n += 1
        if (time.time() - t_check) > 1:
            print("fps : ", f_n) #원래 주석처리 되어 있음 1초에 몇 프레임 찍는지인 거 같음
            t_check = time.time()
            f_n = 0

        draw_img = image.copy()
        lpos, rpos, go = process_image(draw_img)
        print("go", go)

        center = (lpos + rpos) / 2
        angle = -(Width/2 - center + 18) #안쪽은 5로 해봐야할듯

        steer_angle = angle * 0.4
        #draw_steer(steer_angle) #캠 안나오게

        if go==True:
          drive(angle, 19)
        else:
          drive(angle, 0)
            
        cv2.waitKey(1)
        #sq.sleep()

if __name__ == '__main__':
    start()