import cv2
import numpy as np

def detect_faces_and_eyes(image_path=None, use_webcam=False):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process_frame(frame, face_cascade, eye_cascade)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not read image.")
            return
        process_frame(image, face_cascade, eye_cascade)
    
    cv2.destroyAllWindows()

def process_frame(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    cv2.imshow('Face & Eye Detection', frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    detect_faces_and_eyes(use_webcam=True)