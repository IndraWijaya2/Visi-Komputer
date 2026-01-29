import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

MODE = "squat"

KNEE_DOWN, KNEE_UP = 80, 160 

DOWN_R, UP_R = 0.85, 1.00 

SAMPLE_OK = 4 

def ratio_pushup(lm):
    wr = np.array(lm[15][0:2])  
    hp = np.array(lm[23][0:2])  
    
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5, trackCon=0.5)

count, state = 0, "up"
debounce = deque(maxlen=6) 

while True:
    ok, img = cap.read()
    if not ok:
        break
        
    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)
    
    flag = None 

    if lmList:
        if MODE == "squat":
 
            angL, img = detector.findAngle(lmList[23][0:2],
                                            lmList[25][0:2],
                                            lmList[27][0:2],
                                            img=img,
                                            color=(0, 0, 255),
                                            scale=10)
            
            angR, img = detector.findAngle(lmList[24][0:2],
                                            lmList[26][0:2],
                                            lmList[28][0:2],
                                            img=img,
                                            color=(0, 255, 0),
                                            scale=10)
            
            ang = (angL + angR) / 2.0
            
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"
                
            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
        else: 

            r = ratio_pushup(lmList)
            
            if r < DOWN_R:
                flag = "down"
            elif r > UP_R:
                flag = "up"
  
            cv2.putText(img, f"Ratio: {r:4.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        debounce.append(flag)

        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
            
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1
    cv2.putText(img, f"Mode: {MODE.upper()} Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pose Counter", img)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

cap.release()
cv2.destroyAllWindows()