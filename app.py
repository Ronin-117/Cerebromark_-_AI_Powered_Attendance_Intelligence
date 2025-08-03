from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import csv
from datetime import datetime, time, timedelta
import threading
import numpy as np
import time as time_sleep
from apscheduler.schedulers.background import BackgroundScheduler
from hikvisionapi import Client
import requests
from requests.auth import HTTPDigestAuth
import insightface



# ==============================================================================
# --- CONFIGURATION (EDIT ALL SETTINGS HERE) ---
# ==============================================================================

# -- Timetable Configuration --
TIMETABLE = {                           #<------------------------------(FILL YOU TIME TABLE DATA)
    1: (time(8, 45),  time(9,40)),
    2: (time(9, 41),  time(10, 30)),
    3: (time(10,50), time(11, 40)),
    4: (time(11, 41), time(12, 30)),
    5: (time(1, 30), time(2, 20)),
}
VIDEO_RECORDING_DURATION_MINUTES = 5

# -- Hikvision Camera Settings --
CAMERA_IP = "YOUR CAMERA IP"  #<------------------------------(FILL YOU DATA)
USERNAME = "admin" 
PASSWORD = "YOUR PASSWORD"    #<------------------------------(FILL YOU DATA)
CHANNEL_ID_STREAM = 101
CHANNEL_ID_PTZ = 1

# -- Recognition & Database Settings --
MODEL_NAME = "buffalo_l"
RECOGNITION_THRESHOLD = 0.5

# -- NEW: Face Database Path --
DB_PATH = "New_face_db"

# -- PTZ Patrol Settings --
PTZ_ENABLED = True
PTZ_ZOOM_SPEED = 1
PTZ_ZOOM_DURATION_SEC = 17
PTZ_PAUSE_DURATION_SEC = 60*3

# -- File & Directory Paths --
ATTENDANCE_PATH = "attendance_files"
RECORDING_PATH = "recordings"

# -- Display Settings --
UI_DISPLAY_WIDTH = 1280

# ==============================================================================
# --- APPLICATION SETUP ---
# ==============================================================================

app = Flask(__name__)
os.makedirs(ATTENDANCE_PATH, exist_ok=True)
os.makedirs(RECORDING_PATH, exist_ok=True)

face_analysis_app = None
known_faces_db = []
hik_client = None
video_writer = None
ptz_patrol_thread = None
ptz_patrol_active = threading.Event()

recognizer_lock = threading.Lock()
video_lock = threading.Lock()

is_attendance_running = False
current_period = 0
current_attendance_list = []
present_today = set()


def initialize_face_recognition():
    """Loads the insightface model and face database into global variables."""
    global face_analysis_app, known_faces_db
    with recognizer_lock:
        if face_analysis_app is None:
            print("Initializing Face Analysis model...")
            # Use CPUExecutionProvider if you don't have a compatible GPU
            face_analysis_app = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider'])
            face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))
            print("Model loaded.")
        
        print("Loading face database...")
        known_faces_db = load_db()
        print(f"Database loaded with {len(known_faces_db)} known faces.")

def parse_student_info(folder_name):
    try:
        parts = folder_name.split('_', 1)
        return {"Register Number": parts[0], "Name": parts[1]} if len(parts) == 2 else {"Register Number": folder_name, "Name": "Unknown"}
    except Exception:
        return {"Register Number": "Unknown", "Name": "Unknown"}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

def load_db():
    db = []
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database directory '{DB_PATH}' not found.")
        return []
    
    for name in os.listdir(DB_PATH):
        person_dir = os.path.join(DB_PATH, name)
        if not os.path.isdir(person_dir): continue
        for file in os.listdir(person_dir):
            if file.endswith('.npy'):
                try:
                    emb = np.load(os.path.join(person_dir, file))
                    db.append((emb, name))
                except Exception as e:
                    print(f"Could not load file {file}: {e}")
    return db

def find_match(embedding, db):
    if not db: return "Unknown", 0.0
    sims = [cosine_sim(embedding, e) for e, _ in db]
    max_idx = np.argmax(sims)
    max_sim = sims[max_idx]
    if max_sim > RECOGNITION_THRESHOLD:
        return db[max_idx][1], max_sim
    else:
        return "Unknown", max_sim

def send_ptz_zoom(zoom_value):
    if not PTZ_ENABLED: return
    url = f"http://{CAMERA_IP}/ISAPI/PTZCtrl/channels/{CHANNEL_ID_PTZ}/continuous"
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZData><pan>0</pan><tilt>0</tilt><zoom>{zoom_value}</zoom></PTZData>""".strip()
    headers = {"Content-Type": "application/xml"}
    try:
        response = requests.put(
            url, data=payload, headers=headers,
            auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=3)
        if response.status_code != 200: print(f"PTZ Error {response.status_code}: {response.text}")
    except Exception as e: print(f"PTZ Exception: {e}")

def run_ptz_patrol():
    print("PTZ: Zooming in...")
    send_ptz_zoom(PTZ_ZOOM_SPEED)
    time_sleep.sleep(20)
    send_ptz_zoom(0)
    print("PTZ patrol thread started.")
    while ptz_patrol_active.is_set():
        time_sleep.sleep(PTZ_PAUSE_DURATION_SEC)
        print("PTZ: Zooming out...")
        send_ptz_zoom(-PTZ_ZOOM_SPEED)
        time_sleep.sleep(PTZ_ZOOM_DURATION_SEC)
        send_ptz_zoom(0)
        if not ptz_patrol_active.is_set(): break
        print(f"PTZ: Pausing for {PTZ_PAUSE_DURATION_SEC} seconds (zoomed in).")
        time_sleep.sleep(PTZ_PAUSE_DURATION_SEC)
        if not ptz_patrol_active.is_set(): break
        print("PTZ: Zooming in...")
        send_ptz_zoom(PTZ_ZOOM_SPEED)
        time_sleep.sleep(PTZ_ZOOM_DURATION_SEC)
        send_ptz_zoom(0)
        if not ptz_patrol_active.is_set(): break
        print(f"PTZ: Pausing for {PTZ_PAUSE_DURATION_SEC} seconds (zoomed out).")
        time_sleep.sleep(PTZ_PAUSE_DURATION_SEC)
    send_ptz_zoom(0)
    print("PTZ patrol thread stopped.")


def generate_frames():
    global is_attendance_running, current_attendance_list, present_today, video_writer
    
    while is_attendance_running:
        with video_lock:
            if not hik_client:
                print("Hikvision client not available. Stopping frame generation.")
                break
        try:
            # 1. Get frame from camera
            response = hik_client.Streaming.channels[CHANNEL_ID_STREAM].picture(method='get', type='opaque_data')
            image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame_full_res = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if frame_full_res is None: continue

            # 2. Perform recognition using the new direct logic
            faces = face_analysis_app.get(frame_full_res)
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                name_id, similarity = find_match(embedding, known_faces_db)

                if name_id != "Unknown" and name_id not in present_today:
                    present_today.add(name_id)
                    student_info = parse_student_info(name_id)
                    student_info['Status'] = 'Present'
                    current_attendance_list.append(student_info)
                    print(f"Period {current_period}: Marked {student_info['Name']} as Present.")
                
                display_name = parse_student_info(name_id)['Name']
                display_text = f"{display_name} ({similarity:.2f})"
                
                cv2.rectangle(frame_full_res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame_full_res, display_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3. Write frame to video
            with video_lock:
                if video_writer and video_writer.isOpened():
                    video_writer.write(frame_full_res)

            # 4. Resize and yield frame for UI
            h, w = frame_full_res.shape[:2]
            aspect_ratio = h / w
            display_height = int(UI_DISPLAY_WIDTH * aspect_ratio)
            frame_display = cv2.resize(frame_full_res, (UI_DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
            ret, buffer = cv2.imencode('.jpg', frame_display)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation loop: {e}")
            time_sleep.sleep(2)

def start_attendance_session(period):
    global is_attendance_running, current_period, current_attendance_list, present_today, hik_client, video_writer, ptz_patrol_thread
    if is_attendance_running: return
    
    initialize_face_recognition()
    
    print(f"\n--- SCHEDULER: Starting attendance for Period {period} ---")
    is_attendance_running = True
    current_period = period
    current_attendance_list.clear()
    present_today.clear()
    with video_lock:
        try:
            print(f"Connecting to camera at {CAMERA_IP}...")
            hik_client = Client(f'http://{CAMERA_IP}', USERNAME, PASSWORD, timeout=30)
            response = hik_client.Streaming.channels[CHANNEL_ID_STREAM].picture(method='get', type='opaque_data')
            frame = cv2.imdecode(np.asarray(bytearray(response.content), dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w, _ = frame.shape
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            video_filename = f"{timestamp}_period_{period}.mp4"
            video_filepath = os.path.join(RECORDING_PATH, video_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filepath, fourcc, 10.0, (w, h))
            print(f"Started recording {w}x{h} video to {video_filepath}")
            if PTZ_ENABLED:
                ptz_patrol_active.set()
                ptz_patrol_thread = threading.Thread(target=run_ptz_patrol, daemon=True)
                ptz_patrol_thread.start()
        except Exception as e:
            print(f"!!! FATAL ERROR starting session: {e}")
            is_attendance_running = False
            hik_client = None
            video_writer = None

def stop_video_recording():
    global video_writer
    with video_lock:
        if video_writer and video_writer.isOpened():
            video_writer.release()
            video_writer = None
            print(f"--- SCHEDULER: Video recording for period {current_period} finished. ---")


def stop_attendance_session():
    global is_attendance_running, current_period, hik_client, ptz_patrol_thread
    if not is_attendance_running: return
    
    print(f"--- SCHEDULER: Stopping full attendance session for Period {current_period} ---")

    # Stop PTZ patrol
    if PTZ_ENABLED and ptz_patrol_thread:
        print("Stopping PTZ patrol...")
        ptz_patrol_active.clear()
        ptz_patrol_thread.join(timeout=5)
        ptz_patrol_thread = None

    today_date_str = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join(ATTENDANCE_PATH, f"period_{current_period}")
    os.makedirs(folder_path, exist_ok=True)
    fieldnames = ['Register Number', 'Name', 'Status']

    # --- 1. Save PRESENT Students---
    present_filepath = os.path.join(folder_path, f"{today_date_str}_period_{current_period}_present.csv")
    with open(present_filepath, mode='w', newline='', encoding='utf-8') as file:
        if not current_attendance_list:
            file.write("No students were marked present during this session.")
        else:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(current_attendance_list)
    print(f"Present list for Period {current_period} saved to {present_filepath}")

    # --- 2. Calculate and Save ABSENT Students ---
    # Get a set of all student IDs from the database
    all_known_ids = {name_id for _, name_id in known_faces_db}
    absent_ids = all_known_ids - present_today
    
    absentee_list = []
    for name_id in sorted(list(absent_ids)):
        student_info = parse_student_info(name_id)
        student_info['Status'] = 'Absent'
        absentee_list.append(student_info)
        
    absentee_filepath = os.path.join(folder_path, f"{today_date_str}_period_{current_period}_absent.csv")
    with open(absentee_filepath, mode='w', newline='', encoding='utf-8') as file:
        if not absentee_list:
            file.write("All registered students were marked present.")
        else:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(absentee_list)
    print(f"Absentee list for Period {current_period} saved to {absentee_filepath}")
    
    # --- 3. Clean up the session---
    with video_lock:
        if hik_client:
            hik_client = None
            print("Disconnected from Hikvision camera.")
    
    stop_video_recording() 
    is_attendance_running = False
    current_period = 0

# --- Scheduler and Flask Routes ---
scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kolkata')
for period, (start_time, end_time) in TIMETABLE.items():
    scheduler.add_job(start_attendance_session, 'cron', hour=start_time.hour, minute=start_time.minute, args=[period])
    record_stop_time = (datetime.combine(datetime.today(), start_time) + timedelta(minutes=VIDEO_RECORDING_DURATION_MINUTES)).time()
    scheduler.add_job(stop_video_recording, 'cron', hour=record_stop_time.hour, minute=record_stop_time.minute)
    scheduler.add_job(stop_attendance_session, 'cron', hour=end_time.hour, minute=end_time.minute)
scheduler.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not is_attendance_running:
        return "Attendance session is not currently active.", 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_system_status')
def get_system_status():
    return jsonify({"is_running": is_attendance_running, "current_period": current_period, "attendance_data": current_attendance_list})

@app.route('/get_attendance_record', methods=['GET'])
def get_attendance_record():
    date = request.args.get('date')
    period = request.args.get('period')
    if not date or not period: return jsonify({"error": "Missing date or period parameter"}), 400
    filepath = os.path.join(ATTENDANCE_PATH, f"period_{period}", f"{date}_period_{period}.csv")
    if os.path.exists(filepath):
        try:
            with open(filepath, mode='r', newline='', encoding='utf-8') as file:
                first_line = file.readline()
                if "No students were marked present" in first_line: return jsonify([{"message": "No students present."}])
                file.seek(0)
                reader = csv.DictReader(file)
                return jsonify(list(reader))
        except Exception as e: return jsonify({"error": f"Error reading file: {e}"}), 500
    else: return jsonify([])

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask app with Hikvision integration and PTZ patrol.")
    print("Scheduled Jobs:")
    scheduler.print_jobs()
    app.run(debug=False, host='0.0.0.0', threaded=True)