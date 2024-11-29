import cv2
from deepface import DeepFace
from tqdm import tqdm

def write_report(report):
    with open("report.txt", "a") as file:
        file.write(report)
        
        
        
def detect_faces(frame, frame_count, frames_analyzed):
    emotions = DeepFace.analyze(frame, detector_backend="retinaface", actions=['emotion'], enforce_detection=False)
    frames_analyzed += 1
    report = f"\n\nFrame {frame_count}: Quantidade de faces: {len(emotions)}\n"
    write_report(report)

    for index, face in enumerate(emotions):
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        dominant_emotion = face['dominant_emotion']
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        report = f"Face {(index+1)}: Emoçao dominante: {face['dominant_emotion']}\n"
        print(report)
        write_report(report)
    
    return frame, frames_analyzed



def detect_poses(frame, frame_count, frames_analyzed):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        classify_poses(results.pose_landmarks.landmark, mp_pose, frame, frame_count)
    
    return frame, frames_analyzed



def classify_poses(landmarks, mp_pose, frame, frame_count):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    
    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
           
        
    if ( (left_hand.y < left_eye.y) or (left_hand.y < right_eye.y) ):
        report = f"Pose: Mao esquerda levantada\n"
        write_report(report)
    
    if ( (right_hand.y < left_eye.y) or (right_hand.y < right_eye.y) ):
        report = f"Pose: Mao direita levantada\n"
        write_report(report)
    
    if (left_hand.x < left_ear.x):
        report = f"Pose: Mao esquerda no rosto\n"
        write_report(report)
        
    if (right_hand.x > right_ear.x):
        report = f"Pose: Mao direita no rosto\n"
        write_report(report)



def generate_video(out, files):
    for file in tqdm(files):
        frame = cv2.imread(file)
        for _ in range(30):
            out.write(frame)



if __name__ == '__main__':
    files = []
    frames_analyzed = 0

    cap = cv2.VideoCapture("video.mp4")

    if not cap.isOpened():
        print("Ocorreu um erro ao abrir o video.")
        cap.release()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    frame_count = 0

    for _ in tqdm(range(frames)):
        ret, frame = cap.read()

        if not ret:
            break
        

        if (frame_count < 360 or frame_count > 540):
            if (frame_count % 15 == 0):
                frame, frames_analyzed = detect_faces(frame, frame_count, frames_analyzed)
                frame, frames_analyzed = detect_poses(frame, frame_count, frames_analyzed)
                filename = f"output/frame_{frame_count}.jpg"
                files.append(filename)
                cv2.imwrite(filename, frame)

        frame_count += 1
    
    write_report(f"\n\nTotal de frames analisados: {frames_analyzed}\n")
    
    generate_video(out, files)
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()