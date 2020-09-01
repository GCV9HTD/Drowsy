
import cv2
import tensorflow as tf 
import numpy as np
from pygame import mixer

left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

DRIVING = True

# cam = cv2.VideoCapture('http://192.168.0.10:8080/video')
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
model = tf.keras.models.load_model('saved_model/EyeClassify1.3.1')
mixer.init()
sound = mixer.Sound("alarm.wav")
SCORE = 0
MINSIZE = 50
MAXSIZE = 500
MAXSCORE = 20
while True:
    ret, img = cam.read()
    # img = cv2.resize(img,(360,640))
    height,width = img.shape[:2]
    if not ret:
        print("failed to grab img")
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    left_eye = left_eye_cascade.detectMultiScale(gray, scaleFactor=2.4, minNeighbors= 3)
    right_eye = right_eye_cascade.detectMultiScale(gray, scaleFactor=2.4, minNeighbors= 3)
    # right_eye = right_eye_cascade.detectMultiScale(gray,2,10)

    OUT = []
    

    for ex, ey, ew, eh in left_eye:
        left_eye_roi = img[ey:ey+eh, ex:ex+ew]
        l_eye = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (50,50))
        OUT.append(l_eye)
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        
        
    for  x,y,w,h in right_eye:
        right_eye_roi = img[y:y+h, x:x+h]
        r_eye = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (50,50))
        OUT.append(r_eye)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    try:
        OUT = np.array(OUT).reshape(-1,50,50,1)
        state = model.predict(OUT)
        print('left: \t', 'right:' )
        print(state[0][0] ,'\t', state[1][0])
        
        if int(state[0][0]) == 1 or int(state[1][0]):
            cv2.putText(img, "Open", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(img, f"Score:{SCORE}", (10, height-45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            SCORE = SCORE - 1
            if SCORE < 0:
                SCORE = 0
        else:
            cv2.putText(img, f"Score:{SCORE}", (10, height-45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(img, "Closed", (10, height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
            if DRIVING:
                SCORE = SCORE + 1
                if SCORE > MAXSCORE:
                    sound.play(maxtime=50)
    except:
        print('No Eye Found!')
    cv2.imshow("test",img)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv2.destroyAllWindows()