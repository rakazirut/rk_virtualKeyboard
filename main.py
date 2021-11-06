import mediapipe as mp
from cv2 import cv2
import math
from time import sleep
from pynput.keyboard import Controller, Key

frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
finalText = ""
keyboard = Controller()

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils
buttonList = []
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]


def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x+w, y+h), (255, 0, 40), cv2.FILLED)
        cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                    4, (255, 255, 255), 4)
    return img

def getLength (p1, p2):
    x1, y1 = p1.x*frameWidth, p1.y*frameHeight
    x2, y2 = p2.x*frameWidth, p2.y*frameHeight
    length = math.hypot(x2 - x1, y2 - y1)
    return length

class Button():
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text


for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i+50], key))

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    drawAll(img, buttonList)
    if results.multi_hand_landmarks:
        lmList = results.multi_hand_landmarks

        for handLms in lmList:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            bs = getLength(lmList[0].landmark[4], lmList[0].landmark[20])
            fs = getLength(lmList[0].landmark[4], lmList[0].landmark[16])
            
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[0].landmark[8].x*frameWidth < x+w and y < lmList[0].landmark[8].y*frameHeight < y+h:
                cv2.rectangle(img, (x-5,y-5), (x+w+5, y+h+5),
                                (255, 140, 0), cv2.FILLED)
                cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                            4, (255, 255, 255), 4)

                dist = getLength(lmList[0].landmark[8], lmList[0].landmark[12])
                # When click
                if dist<30:
                    keyboard.press(button.text)
                    keyboard.release(button.text)
                    print('{} registered'.format(button.text))
                    cv2.rectangle(img, button.pos, (x+w, y+h),
                                    (140, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)
                    finalText += button.text
                    sleep(0.10)

                if bs<20:
                    keyboard.press(Key.backspace)
                    keyboard.release(Key.backspace)
                    print('backspace registered')
                    finalText = finalText[:-1]
                    sleep(0.10)

                if fs<20:
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
                    print('space registered')
                    finalText += " "
                    sleep(0.10)

    cv2.rectangle(img, (50,350), (700,450),
                                    (255, 0, 40), cv2.FILLED)
    cv2.putText(img, finalText, (60, 420), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
