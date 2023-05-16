import cv2  
import numpy as np  
import os  
import tensorflow as tf  
from mtcnn import MTCNN  
class LipReading:  
    def __init__(self, model_path):  
        self.detector = MTCNN() 
        self.graph = tf.Graph()  
        with self.graph.as_default():  
            self.sess = tf.Session()  
            saver = tf.train.import_meta_graph(model_path + '/model.meta')  
            saver.restore(self.sess, tf.train.latest_checkpoint(model_path))  
            self.X = self.graph.get_tensor_by_name("X:0")  
            self.y_pred = self.graph.get_tensor_by_name("y_pred:0")  
            self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")  
    def extract_features(self, video_path):  
        video = cv2.VideoCapture(video_path)  
        frames = []  
        while True:  
            ret, frame = video.read()  
            if not ret:  
                break  
            frames.append(frame)  
        video.release()  
        features = []  
        for frame in frames:  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            faces = self.detector.detect_faces(frame) 
            if len(faces) == 0:  
                continue  
            x1, y1, w, h = faces[0]['box'] 
            x2, y2 = x1 + w, y1 + h 
            mouth = gray[y1:y2, x1:x2]  
            mouth = cv2.resize(mouth, (48, 48))  
            mouth = np.asarray(mouth).reshape(1, 48, 48, 1)  
            with self.graph.as_default():  
                pred = self.sess.run(self.y_pred, feed_dict={self.X: mouth, self.keep_prob: 1.0})  
            features.append(pred[0])  
        return np.asarray(features) 
    def extract_features_realtime(self): 
        cap = cv2.VideoCapture(0) 
        while True: 
            ret, frame = cap.read() 
            if not ret: 
                print("Error: Failed to capture frame from camera.") 
                break 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            faces = self.detector.detect_faces(frame) 
            if len(faces) == 0: 
                continue  
            x1, y1, w, h = faces[0]['box'] 
            x2, y2 = x1 + w, y1 + h 
            mouth = gray[y1:y2, x1:x2]  
            mouth = cv2.resize(mouth, (48, 48)) 
            mouth = np.asarray(mouth).reshape(1, 48, 48, 1) 
            with self.graph.as_default(): 
                pred = self.sess.run(self.y_pred, feed_dict={self.X: mouth, self.keep_prob: 1.0}) 
            cv2.imshow('Lip Reading', frame) 
            if cv2.waitKey(1) == ord('q'): 
                break 
        cap.release() 
        cv2.destroyAllWindows()  
    def extract_features_from_directory(self, directory_path): 
        features = [] 
        for filename in os.listdir(directory_path): 
            if filename.endswith(".mp4"): 
                filepath = os.path.join(directory_path, filename) 
                video_features = self.extract_features(filepath) 
                features.append(video_features) 
        return np.concatenate(features, axis=0)
