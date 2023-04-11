import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
from urllib.request import urlopen
import numpy as np
import socket
import time
import pandas as pd


def esp_buzzer():
    esp32_ip = '192.168.250.58'
    esp32_port = 1234

    # Create a socket connection to the ESP32
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((esp32_ip, esp32_port))

    i=0
    while(i<10):
        message = 'Hello\n'
        sock.sendall(message.encode())
        i+=1

    
    sock.close()



# Face recognition and opencv setup
face_detector = dlib.get_frontal_face_detector()
Emp = pd.DataFrame()
tm = []
drw = []

def Detect_Eye(eye):
        poi_A = distance.euclidean(eye[1], eye[5])
        poi_B = distance.euclidean(eye[2], eye[4])
        poi_C = distance.euclidean(eye[0], eye[3])
        aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
        return aspect_ratio_Eye


dlib_facelandmark = dlib.shape_predictor("F:\\TARP\\shape_predictor_68_face_landmarks.dat")
url='http://192.168.137.33/capture'
while True:
    
    img_resp = urlopen(url)
    imgnp = np.asarray(bytearray(img_resp.read()), dtype="uint8")
    img = cv2.imdecode(imgnp, -1)
    # cv2.imshow("Camera", img)
    engine = pyttsx3.init()

 
    
        # null, frame = img_resp.read()
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_detector(gray_scale)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []
 
        
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)
 
       
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (255, 255, 0), 1)
 
        
        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye+right_Eye)/2
 
        
        Eye_Rat = round(Eye_Rat, 2)
 
        # THIS VALUE OF 0.25 (YOU CAN EVEN CHANGE IT)
        # WILL DECIDE WHETHER THE PERSONS'S EYES ARE CLOSE OR NOT

        if Eye_Rat < 0.25:
            cv2.putText(img, "DROWSINESS DETECTED", (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
            cv2.putText(img, "Alert!!!! WAKE UP DUDE", (50, 450),
                    cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
 
            t = time.localtime()
            # esp_buzzer()
            engine.say("Alert!!!! WAKE UP DUDE")
            engine.runAndWait()
            current_time = time.strftime("%H:%M", t)
            tm.append(current_time)
            drw.append(1)
 
    cv2.imshow("Drowsiness DETECTOR IN OPENCV2", img)
    key = cv2.waitKey(9)
    if cv2.waitKey(1) == 113:
        break

print("hai")
Emp['Time']=tm
Emp['Drowsy'] = drw

Emp.to_csv(r'F:\TARP\res.csv')




