import mediapipe as mp
from cv2 import cv2

frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
count = 0
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

        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[0].landmark[8].x*frameWidth < x+w and y < lmList[0].landmark[8].y*frameHeight < y+h:
                    cv2.rectangle(img, button.pos, (x+w, y+h),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
