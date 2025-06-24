from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
import threading
import os
import csv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

# Global variables
camera = None
is_camera_running = False
frame_lock = threading.Lock()
known_face_encodings = []
known_face_names = []
marked_attendance = {}

# Email Configuration
SENDER_EMAIL = "sender-email"
SENDER_PASSWORD = "sender-password"
RECEIVER_EMAIL = "receiver-email"

# Database setup
DATABASE_PATH = "attendance.db"

def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            registration_date TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            date TEXT,
            time TEXT,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()

init_database()

def load_known_faces():
    """Load known faces from the `Images` directory."""
    global known_face_encodings, known_face_names
    image_dir = "Images"
    known_face_names = []
    known_face_encodings = []

    for file_name in os.listdir(image_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            name = os.path.splitext(file_name)[0].replace("_", " ")
            file_path = os.path.join(image_dir, file_name)

            try:
                image = face_recognition.load_image_file(file_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                # Add to database
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO students (name, registration_date) VALUES (?, ?)",
                               (name, datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error loading face for {name}: {e}")

load_known_faces()

def mark_attendance(name):
    """Mark attendance for a detected person."""
    global marked_attendance
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    if name not in marked_attendance:
        marked_attendance[name] = {"time": current_time, "date": current_date}

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Get the student ID
        cursor.execute("SELECT id FROM students WHERE name=?", (name,))
        student_id = cursor.fetchone()[0]

        # Insert attendance record
        cursor.execute("""
            INSERT INTO attendance (student_id, date, time)
            VALUES (?, ?, ?)
        """, (student_id, current_date, current_time))
        conn.commit()
        conn.close()
        print(f"Attendance marked for {name} at {current_time}")

def generate_frame():
    """Video frame generator for the live feed."""
    global camera, is_camera_running
    while True:
        with frame_lock:
            # Exit the loop if the camera is no longer running
            if not is_camera_running or camera is None:
                break

            success, frame = camera.read()
            if not success:
                continue

            # Detect and recognize faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]
                    mark_attendance(name)

                # Draw rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['GET'])
def start_camera():
    """Start the video feed."""
    global camera, is_camera_running
    if not is_camera_running:
        camera = cv2.VideoCapture(0)
        is_camera_running = True
    return jsonify({"status": "Camera started."})

@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    """Stop the video feed."""
    global camera, is_camera_running
    with frame_lock:  # Ensure thread safety
        if is_camera_running:
            if camera and camera.isOpened():  # Check if the camera is valid and open
                camera.release()  # Release the camera safely
            camera = None  # Reset the camera to None
            is_camera_running = False
            return jsonify({"status": "Camera stopped successfully."})
        else:
            return jsonify({"status": "Camera is not running."})  # Handle already stopped state


@app.route('/export_csv', methods=['GET'])
def export_csv():
    """Export attendance to a CSV file."""
    if not marked_attendance:
        return jsonify({"error": "No attendance to export."})

    csv_file_path = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Name", "Time", "Date"])
        for name, data in marked_attendance.items():
            writer.writerow([name, data["time"], data["date"]])

    return send_file(csv_file_path, as_attachment=True)

@app.route('/send_email', methods=['POST'])
def send_email():
    """Send attendance report via email."""
    try:
        # Get the receiver's email from the POST request
        data = request.get_json()
        receiver_email = data.get('receiver_email')

        if not receiver_email:
            return jsonify({"error": "Receiver email not provided."})

        csv_file_path = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Name", "Time", "Date"])
            for name, data in marked_attendance.items():
                writer.writerow([name, data["time"], data["date"]])

        # Setup email
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = receiver_email
        message['Subject'] = "Attendance Report"

        body = "Please find the attached attendance report."
        message.attach(MIMEText(body, 'plain'))

        with open(csv_file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(csv_file_path)}')
        message.attach(part)

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)

        return jsonify({"status": "Email sent successfully."})

    except Exception as e:
        return jsonify({"error": f"Failed to send email: {e}"})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
