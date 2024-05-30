import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model and labels dictionary
with open('./dataset_processed_all_data.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change 0 to 2 if you want to use a different camera

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Function to pad sequences to the same length as training data
def pad_sequence(sequence, max_length, pad_value=0):
    return sequence + [pad_value] * (max_length - len(sequence))

# Determine the padding length from the training data (ensure this matches the training script)
padding_length = 84  # This should be set to the length used in the training data

# Label dictionary (update based on your model's training labels)
# labels_dict = {i - 64: chr(i) for i in range(65, 91)}  # {'A': 1, 'B': 2, ..., 'Z': 26}
labels_dict = {0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 'I', 9 : 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R', 18 : 'S', 19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if data_aux:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Pad the data_aux to the required length
            padded_data_aux = pad_sequence(data_aux, padding_length)
            
            # Convert to the required format for the model
            padded_data_aux = np.array(padded_data_aux).reshape(1, -1)

            prediction = model.predict(padded_data_aux)
            index = int(prediction[0])
            if index in labels_dict:
                predicted_character = labels_dict[index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            else:
                print('nothing')

            
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
