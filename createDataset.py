import os 
import pickle 
import mediapipe as mp 
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './trainset'
dataset = []
labels = []



for dir in os.listdir(data_dir):
    dir_path = os.path.join(data_dir, dir)
    
    # Ensure the item is a directory
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img_file_path = os.path.join(dir_path, img_path)
            
            # Ensure the item is a file and not .DS_Store
            if os.path.isfile(img_file_path) and img_path != '.DS_Store':
                temp_data = []
                x_ = []
                y_ = []

                img = cv2.imread(img_file_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            temp_data.append(x - min(x_))
                            temp_data.append(y - min(y_))

                    dataset.append(temp_data)
                    labels.append(dir)


file = open('dataset_processed_5.pickle', 'wb')
pickle.dump({'data': dataset, 'labels': labels}, file)
file.close()
