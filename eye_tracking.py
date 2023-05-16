import numpy as np 
import cv2 
import dlib 
from scipy.spatial import distance as dist 
import speech_recognition as sr 
import threading 
import deepface 
from deepface import DeepFace 
from tkinter import * 
from datetime import datetime 
 # Load pre-trained facial landmark detector 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
 # Function for calculating Eye Aspect Ratio 
def eye_aspect_ratio(eye): 
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4]) 
    C = dist.euclidean(eye[0], eye[3]) 
    ear = (A + B) / (2.0 * C) 
    return ear 
 # Initialize video stream 
cap = cv2.VideoCapture(0) 
 # Initialize text-to-speech and speech-to-text modules 
r = sr.Recognizer() 
with sr.Microphone() as source: 
    print("Speak now...") 
    audio = r.listen(source) 
 def speech_recognition(): 
    try: 
        print("You said: " + r.recognize_google(audio)) 
        return r.recognize_google(audio) 
    except sr.UnknownValueError: 
        print("Unable to recognize speech") 
    except sr.RequestError as e: 
        print("Error: " + str(e)) 
 # Function for detecting emotions in facial expressions 
def detect_emotions(frame): 
    emotions = ["happy", "surprise", "neutral", "sad", "angry", "fear"] 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    result = DeepFace.analyze(img, actions=emotions) 
    return result 
 # Function for calculating gaze direction 
def pupil_distance(eye): 
    A = dist.euclidean(eye[0], eye[3]) 
    return A 
 def get_gaze_direction(frame, landmarks): 
    eye_left = landmarks[36:42] 
    eye_right = landmarks[42:48] 
     eye_left = np.array(eye_left) 
    eye_right = np.array(eye_right) 
     d_left = pupil_distance(eye_left) 
    d_right = pupil_distance(eye_right) 
     eye_points_left = [] 
    eye_points_right = [] 
     for i in range(36, 42): 
        x = landmarks[i].x 
        y = landmarks[i].y 
        eye_points_left.append((x, y)) 
     for i in range(42, 48): 
        x = landmarks[i].x 
        y = landmarks[i].y 
        eye_points_right.append((x, y)) 
     eye_points_left = np.array(eye_points_left, dtype=np.int32) 
    eye_points_right = np.array(eye_points_right, dtype=np.int32) 
     min_left_x, min_left_y, min_left_w, min_left_h = cv2.boundingRect(eye_points_left) 
    min_right_x, min_right_y, min_right_w, min_right_h = cv2.boundingRect(eye_points_right) 
     roi_left = frame[min_left_y: min_left_y + min_left_h, min_left_x: min_left_x + min_left_w] 
    roi_right = frame[min_right_y: min_right_y + min_right_h, min_right_x: min_right_x + min_right_w] 
     try: 
        gray_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY) 
        thresh1 = cv2.threshold(gray_left, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 
        thresh1 = cv2.erode(thresh1, None, iterations=2) 
        thresh1 = cv2.dilate(thresh1, None, iterations=4) 
        thresh1 = cv2.medianBlur(thresh1, 5) 
    except: 
        print('Error in roi_left') 
     try: 
        gray_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY) 
        thresh2 = cv2.threshold(gray_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 
        thresh2 = cv2.erode(thresh2, None, iterations=2) 
        thresh2 = cv2.dilate(thresh2, None, iterations=4) 
        thresh2 = cv2.medianBlur(thresh2, 5) 
    except: 
        print('Error in roi_right') 
     _, contours1, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    _, contours2, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
     try: 
        cnt_left = max(contours1, key=cv2.contourArea) 
        M_left = cv2.moments(cnt_left) 
        cx_left = int(M_left['m10'] / M_left['m00']) 
        cy_left = int(M_left['m01'] / M_left['m00']) 
        center_left = (cx_left, cy_left) 
        cv2.circle(roi_left, center_left, 5, (0, 0, 255), -1) 
    except: 
        print('Error in contours1') 
     try: 
        cnt_right = max(contours2, key=cv2.contourArea) 
        M_right = cv2.moments(cnt_right) 
        cx_right = int(M_right['m10'] / M_right['m00']) 
        cy_right = int(M_right['m01'] / M_right['m00']) 
        center_right = (cx_right, cy_right) 
        cv2.circle(roi_right, center_right, 5, (0, 0, 255), -1) 
    except: 
        print('Error in contours2') 
     try: 
        if cx_left < cx_right: 
            return "right" 
        else: 
            return "left" 
    except: 
        print('Error in cx_left and cx_right') 
 # Function for displaying the gaze direction and emotion 
def display_result(frame, gaze_direction, emotion): 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1 
    color = (255, 0, 0) 
    thickness = 2 
    cv2.putText(frame, "Gaze Direction: " + str(gaze_direction), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    cv2.putText(frame, "Emotion: " + str(emotion), (50, 100), font, fontScale, color, thickness, cv2.LINE_AA) 
 def log_data(gaze_direction, emotion, speech): 
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
    with open("log.txt", "a+") as file: 
        file.write(current_time + "," + str(gaze_direction) + "," + str(emotion) + "," + str(speech) + "\n") 
 
while True:  
    # Read video frame  
    ret, frame = cap.read()  
 
    # Detect faces  
    faces = detector(frame)  
 
    # Loop through detected faces  
    for face in faces:  
        landmarks = predictor(frame, face)  
 
        # Calculate gaze direction  
        gaze_direction = get_gaze_direction(frame, landmarks)  
 
        # Calculate emotions  
        emotion_result = detect_emotions(frame)  
        emotion = max(emotion_result, key=emotion_result.get)  
 
        # Display gaze direction and emotion on screen  
        display_result(frame, gaze_direction, emotion)  
 
        # Start speech recognition in a separate thread  
        speech_thread = threading.Thread(target=speech_recognition, args=())  
        speech_thread.start()  
 
        # Get speech result  
        speech = speech_recognition()  
 
        # Log gaze direction, emotion, and speech  
        log_data(gaze_direction, emotion, speech)  
 
    # Display the resulting frame  
    cv2.imshow('Frame', frame)  
 
    # Exit loop if 'q' is pressed  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
 
# Release the video capture object and close windows  
cap.release()  
cv2.destroyAllWindows()
