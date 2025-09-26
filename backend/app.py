from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)

bcrypt = Bcrypt(app)
app.config["JWT_SECRET_KEY"] = "kuch_bhi_secret_jo_complex_ho"
jwt = JWTManager(app)

users = []

# Load students data from students.json
try:
    with open("students.json", "r") as f:
        students = json.load(f)
    print(f"[INFO] Loaded {len(students)} students from database")
except FileNotFoundError:
    print("[ERROR] students.json not found!")
    students = []

known_encodings = []
known_names = []

print("[INFO] Encoding known faces...")
for student in students:
    image_path = os.path.join("images", student["image"])
    if os.path.exists(image_path):
        try:
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student["name"])
                print(f"[SUCCESS] Encoded {student['name']}")
            else:
                print(f"[WARNING] No face found in {student['name']}'s image")
        except Exception as e:
            print(f"[ERROR] Failed to encode {student['name']}: {e}")
    else:
        print(f"[ERROR] Image not found: {image_path}")

print(f"[INFO] Successfully encoded {len(known_encodings)} faces")

@app.route("/", methods=["GET"])
def home():
    return """
    <h1>SnapTick Face Recognition API</h1>
    <p>Upload a photo to /recognize endpoint</p>
    <form action="/recognize" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Recognize Faces</button>
    </form>
    """

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    if not all([name, email, password, role]):
        return jsonify({"error": "Missing fields"}), 400

    if any(u["email"] == email for u in users):
        return jsonify({"error": "Email already registered"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    users.append({"name": name, "email": email, "password": hashed_password, "role": role})

    return jsonify({"msg": "Signup successful"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = next((u for u in users if u["email"] == email), None)
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    if not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    access_token = create_access_token(identity={"email": user["email"], "role": user["role"], "name": user["name"]})
    return jsonify({"access_token": access_token}), 200

@app.route("/recognize", methods=["POST"])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img = face_recognition.load_image_file(file)
        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)

        print(f"[INFO] Found {len(encodings)} faces in uploaded image")

        present = []
        unknown = []
        detected_names = set()

        for enc in encodings:
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, enc)
            best_match_index = None if len(distances) == 0 else distances.argmin()
            name = "Unknown"

            if True in matches and best_match_index is not None and distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
                detected_names.add(name)
                if name not in present:
                    present.append(name)
                print(f"[MATCH] Recognized: {name}")
            else:
                unknown.append("Unknown student detected")
                print("[UNKNOWN] Face not recognized")

        absent = [s["name"] for s in students if s["name"] not in detected_names]

        result = {
            "success": True,
            "total_faces_detected": len(encodings),
            "present": present,
            "absent": absent,
            "unknown": len(unknown),
            "timestamp": str(cv2.getTickCount())
        }

        print(f"[RESULT] Present: {present}, Absent: {absent}")
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/students", methods=["GET"])
def get_students():
    return jsonify(students)

if __name__ == "__main__":
    if len(known_encodings) == 0:
        print("[WARNING] No faces encoded! Make sure images are in 'images/' folder")
    print(f"[INFO] Starting server with {len(known_encodings)} known faces...")
    app.run(host='0.0.0.0', port=5000, debug=True)
