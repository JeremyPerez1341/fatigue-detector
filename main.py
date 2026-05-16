import cv2
import mediapipe as mp
import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    
    if horizontal == 0:
        return 0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calculate_mar(mouth):
    upper_lip, lower_lip, left_corner, right_corner = mouth

    vertical = euclidean_distance(upper_lip, lower_lip)
    horizontal = euclidean_distance(left_corner, right_corner)
    
    if horizontal == 0:
        return 0

    mar = vertical / horizontal
    return mar

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.20
MIN_EAR = 0.15
CLOSED_EYES_FRAMES = 0
FATIGUE_FRAMES = 15
MOUTH = [13, 14, 78, 308]
MAR_THRESHOLD = 0.24
MOUTH_OPEN_FRAMES = 0
YAWN_FRAMES = 15

# Access the module that detects faces with points
mp_face_mesh = mp.solutions.face_mesh

# Create the detector
face_mesh = mp_face_mesh.FaceMesh()

# Draw lines, points and visualize results
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame -> "ret": successful reading | "frame": captured image
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # face_landmarks => +-468 points
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            left_eye_coords = []
            for idx in LEFT_EYE:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                left_eye_coords.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            right_eye_coords = []
            for idx in RIGHT_EYE:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                right_eye_coords.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            left_ear = calculate_ear(left_eye_coords)
            right_ear = calculate_ear(right_eye_coords)
            
            left_valid = left_ear > MIN_EAR
            right_valid = right_ear > MIN_EAR

            if left_valid and right_valid:
                ear = (left_ear + right_ear) / 2.0
            elif left_valid:
                ear = left_ear
            elif right_valid:
                ear = right_ear
            else:
                ear = 1  # avoid false positive
                
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if ear < EAR_THRESHOLD:
                CLOSED_EYES_FRAMES += 1
                cv2.putText(frame, "OJOS CERRADOS", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                CLOSED_EYES_FRAMES = 0
                cv2.putText(frame, "OJOS ABIERTOS", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if CLOSED_EYES_FRAMES >= FATIGUE_FRAMES:
                cv2.putText(frame, "FATIGA DETECTADA", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            
            mouth_coords = []

            for idx in MOUTH:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                mouth_coords.append((x, y))

                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
            mar = calculate_mar(mouth_coords)
            
            cv2.putText(frame, f"MOUTH FRAMES: {MOUTH_OPEN_FRAMES}", (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if mar > MAR_THRESHOLD:
                MOUTH_OPEN_FRAMES += 1
            else:
                MOUTH_OPEN_FRAMES = 0
            
            if MOUTH_OPEN_FRAMES >= YAWN_FRAMES:
                cv2.putText(frame, "BOSTEZO DETECTADO", (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Open a window
    cv2.imshow("Deteccion Facial", frame)

    # Escape with "ESC"
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Free up resources
cap.release()
cv2.destroyAllWindows()