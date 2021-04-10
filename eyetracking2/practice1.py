from time import sleep

import cv2
import numpy as np
import dlib
from math import hypot
import math

""" determine mid point
"""
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


""" Detect eye's location
    and blinking
"""
def get_blinking_ratio(facial_landmarks):
    left_point1 = (facial_landmarks.part(36).x, facial_landmarks.part(36).y)
    right_point1 = (facial_landmarks.part(39).x, facial_landmarks.part(39).y)
    center_top1 = midpoint(facial_landmarks.part(37), facial_landmarks.part(38))
    center_bottom1 = midpoint(facial_landmarks.part(41), facial_landmarks.part(40))

    left_point2 = (facial_landmarks.part(42).x, facial_landmarks.part(42).y)
    right_point2 = (facial_landmarks.part(45).x, facial_landmarks.part(45).y)
    center_top2 = midpoint(facial_landmarks.part(43), facial_landmarks.part(44))
    center_bottom2 = midpoint(facial_landmarks.part(47), facial_landmarks.part(46))

    # cv2.line(frame, left_point1, right_point1, (0, 255, 0), 1)      # hor line 1
    # cv2.line(frame, center_top1, center_bottom1, (0, 255, 0), 1)    # ver line 1
    # cv2.line(frame, left_point2, right_point2, (0, 255, 0), 1)      # hor line 2
    # cv2.line(frame, center_top2, center_bottom2, (0, 255, 0), 1)    # ver line 2

    ver_line_len1 = hypot((center_top1[0] - center_bottom1[0]), (center_top1[1] - center_bottom1[1]))
    hor_line_len1 = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))
    ver_line_len2 = hypot((center_top2[0] - center_bottom2[0]), (center_top2[1] - center_bottom2[1]))
    hor_line_len2 = hypot((left_point2[0] - right_point2[0]), (left_point2[1] - right_point2[1]))

    blink_ratio_left = hor_line_len1 / ver_line_len1
    blink_ratio_right = hor_line_len2 / ver_line_len2
    blink_ratio = (blink_ratio_left + blink_ratio_right) / 2

    return blink_ratio


""" Detect eye's gazing
"""
def get_gaze_ratio(eye_points, facial_landmarks):

    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    cv2.polylines(frame, [eye_region], True, (0, 255, 255), 1)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 1)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])    

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # 눈동자의 흰부분 계산으로 보는 방향 추정
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    cv2.imshow("left", left_side_threshold)
    cv2.imshow("right", right_side_threshold)

    # left, right side white 가 10 미만이면 눈 감은것으로 인식
    if left_side_white < 10 or right_side_white < 10:
        _gaze_ratio = 1
    else:
        _gaze_ratio = left_side_white / right_side_white

    return _gaze_ratio




"""""""""""""""""""""""""""""""""""""""
""""""     # MAIN FUNCTION #     """"""
"""""""""""""""""""""""""""""""""""""""
cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # 네모 감지 부분



    # 얼굴 인식 부분
    for face in faces:
        landmarks = predictor(gray, face)


        # 눈을 추적하며 깜박임 감지
        if get_blinking_ratio(landmarks) > 3.7:   # 숫자가 높아질수록 엄격하게 감지
            cv2.putText(frame, "Blink", (50, 150), font, 3, (255, 0, 0))


        # 보는 방향 감지
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        '''
        if gaze_ratio < 0.75:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
        elif 0.75 < gaze_ratio < 1.1:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
        '''

        # 숫자가 작아질수록 관대
        if gaze_ratio < 0.55:
            print("눈동자 오른쪽으로 벗어남\n")

        # 숫자가 커질수록 관대
        elif gaze_ratio > 2.2:
            print("눈동자 왼쪽으로 벗어남\n")

    #Print ont the screen
    cv2.imshow("Frame", frame)

    #Press ESC to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()