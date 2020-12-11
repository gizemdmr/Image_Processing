import numpy as np
import cv2
import csv
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import db_connect

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.show()

def ornek(video):
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video)
    # -----------------------------
    # face expression recognizer initialization
    from tensorflow.keras.models import model_from_json
    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5')  # load weights
    # -----------------------------
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    count = 0
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            detected_face1 = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
            detected_face = cv2.cvtColor(detected_face1, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
            predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
            # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            max_index = np.argmax(predictions[0])
            emotion = emotions[max_index]
            # write emotion text above rectangle
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # process on detected face end
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        count += 1
        if key == ord('s'):
            if (video == 'test_image/video1.mov'):
                filename = 'saved_img' + str(count) + '.jpg'
                path = 'test_image/in/' + filename
                cv2.imwrite(path, img=detected_face1)
                db_connect.baglan(path, video, emotion)
            if (video == 'test_image/video2.mov'):
                filename = 'saved_img' + str(count) + '.jpg'
                path = 'test_image/out/' + filename
                cv2.imwrite(path, img=detected_face1)
                db_connect.baglan(path, video, emotion)

        if key == ord('q'):  # press q to quit
            break

        # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


