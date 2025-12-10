import numpy as np
import cv2
import mediapipe as mp
import pygame
pygame.mixer.init()
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6, #if the code doesn't detect the face try incresing this a little not too much
    min_tracking_confidence=0.6 #same here
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1) #Use one of the numbers that you get from cam_test.py or a video file
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No camera available!")
        exit(1)

#Eye landmark indices form Mediapipe
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_EAR(eye_points, face_landmarks, frame_width, frame_height): #for calculating the EAR
    # eye_points = list of 6 landmark indices
    points = []
    for idx in eye_points:
        lm = face_landmarks.landmark[idx]
        points.append((int(lm.x * frame_width), int(lm.y * frame_height)))
#face detected
    # vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    EAR = (A + B) / (2.0 * C)
    #EAR = Eye Aspect Ratio
    return EAR 
    #Notes:
    #EAR ≈ 0.25–0.30 eyes open
    #EAR ≈ 0.10 or less eyes closed

#NEW: Calibration parameters 
CALIBRATE_SECONDS = 2.0   # how long to measure baseline at start (seconds)
calib_ear_values = []
calibrated = False
calib_start_time = None

#you can play with these numbers a little bit to make it strict or not
EAR_THRESHOLD = 0.20     # below this -> eyes likely closed 
FRAMES_THRESHOLD = 25    # how many frames before triggering alarm
fps = cap.get(cv2.CAP_PROP_FPS)

# for counting the frames with closed eyes
counter = 0 
alarm_on = False

# new: for EAR smoothing
ear_history = []
EAR_SMOOTH_FRAMES = 7  # increased smoothing window
face_missing_counter = 0

# 3D model points (fixed reference from Mediapipe)
FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),# Left eye left corner
    (225.0, 170.0, -135.0), # Right eye right corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)

# Run a short calibration phase to get a personalized baseline EAR
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0  # fallback if video file doesn't report FPS reliably

calib_frame_limit = int(max(1, CALIBRATE_SECONDS * fps))
calib_frames_collected = 0

# store timestamp to stop calibration after time even if face not detected often
calib_start_wall = time.time()

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # Convert BGR(OpenCV default) to RGB(for Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w = frame.shape[:2]

    # If face detected
    if results.multi_face_landmarks:
        face_missing_counter = 0
        for face_landmarks in results.multi_face_landmarks:

            # 2D facial landmark points for pose estimation
            FACE_2D_POINTS = np.array([
                (face_landmarks.landmark[1].x * frame_w, face_landmarks.landmark[1].y * frame_h),   # Nose tip
                (face_landmarks.landmark[152].x * frame_w, face_landmarks.landmark[152].y * frame_h), # Chin
                (face_landmarks.landmark[263].x * frame_w, face_landmarks.landmark[263].y * frame_h), # Left eye right corner
                (face_landmarks.landmark[33].x * frame_w, face_landmarks.landmark[33].y * frame_h),   # Right eye left corner
                (face_landmarks.landmark[287].x * frame_w, face_landmarks.landmark[287].y * frame_h), # Left mouth corner
                (face_landmarks.landmark[57].x * frame_w, face_landmarks.landmark[57].y * frame_h)    # Right mouth corner
            ], dtype=np.float64)

            # Camera matrix (approximation)
            focal_length = frame_w
            cam_matrix = np.array([[focal_length, 0, frame_w / 2],
                                   [0, focal_length, frame_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Head pose estimation
            success, rot_vec, trans_vec = cv2.solvePnP(FACE_3D_POINTS, FACE_2D_POINTS, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360

            # Calculate EAR
            left_EAR = calculate_EAR(LEFT_EYE, face_landmarks, frame_w, frame_h)
            right_EAR = calculate_EAR(RIGHT_EYE, face_landmarks, frame_w, frame_h)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # If not yet calibrated, collect baseline EAR values for CALIBRATE_SECONDS
            if not calibrated:
                if calib_start_time is None:
                    calib_start_time = time.time()
                # collect limited number of frames
                if calib_frames_collected < calib_frame_limit and (time.time() - calib_start_wall) < (CALIBRATE_SECONDS + 1.0):
                    calib_ear_values.append(avg_EAR)
                    calib_frames_collected += 1
                # finish calibration either by frames or by time
                if calib_frames_collected >= calib_frame_limit or (time.time() - calib_start_time) >= CALIBRATE_SECONDS:
                    if len(calib_ear_values) > 0:
                        baseline_max_EAR = max(calib_ear_values)
                        # personalized threshold: fraction of baseline, but not lower than 0.17
                        EAR_THRESHOLD = max(0.17, 0.55 * baseline_max_EAR)
                        # increase frames threshold slightly if baseline indicates slow blinks
                        FRAMES_THRESHOLD = 35
                    calibrated = True
            baseline_avg_EAR = np.mean(calib_ear_values)  # for adaptive drop detection

            # new: smooth EAR values
            ear_history.append(avg_EAR)
            if len(ear_history) > EAR_SMOOTH_FRAMES:
                ear_history.pop(0)
            smooth_EAR = np.mean(ear_history)

            # Draw eyes only
            for lm_id in LEFT_EYE:
                lm = face_landmarks.landmark[lm_id]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # red dots = left eye

            for lm_id in RIGHT_EYE:
                lm = face_landmarks.landmark[lm_id]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # blue dots = right eye

            #Drowsiness + head movement detection
            head_down = pitch > 30 or pitch < -30  # less strict: wider tolerance
            head_side = yaw > 55 or yaw < -55      # less strict: wider tolerance

            # Check if eyes are closed enough OR dropped sharply from recent average
            EAR_DROP_RATIO = 0.75  # 25% drop triggers
            recent_avg_EAR = np.mean(ear_history[-EAR_SMOOTH_FRAMES:])

            if (smooth_EAR < EAR_THRESHOLD or smooth_EAR < baseline_avg_EAR * EAR_DROP_RATIO) and (head_down or head_side):
                counter += 1
                if counter >= fps:
                    cv2.putText(frame, "DROWSINESS + HEAD MOVEMENT ALERT!", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    # play sound until it ends
                    if not alarm_on:    
                        #change to the alarm file you have
                        pygame.mixer.music.load(r"C:\Users\User\Desktop\Projects\Drowsiness-Detection\assets\Alarm.mp3")  # your .wav/.mp3 file
                        pygame.mixer.music.play()
                        alarm_on = True
                    elif alarm_on and not pygame.mixer.music.get_busy():
                        #if the alarm finishes before the person wakes up play it again and again anddd againnnn
                        pygame.mixer.music.play()
                    
            else:
                counter = 0
                if alarm_on:
                    if not pygame.mixer.music.get_busy():
                        alarm_on = False

            # Show EAR and head angles
            cv2.putText(frame, f"EAR: {smooth_EAR:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Pitch: {int(pitch)}  Yaw: {int(yaw)}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # show calibration status (optional small indicator)
            if not calibrated:
                cv2.putText(frame, f"Calibrating... ({calib_frames_collected}/{calib_frame_limit})", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        face_missing_counter += 1
        if face_missing_counter > fps:
            cv2.putText(frame, "Warning: Face not detected! Lighting too dark?", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)
    
        
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
#don't forget theseee or you will destory you Ram and GPU
cap.release()
cv2.destroyAllWindows()