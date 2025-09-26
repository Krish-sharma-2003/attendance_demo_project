import face_recognition
import os
import pickle

# Known faces directory
KNOWN_FACES_DIR = "backend/images"
ENCODINGS_PATH = "backend/encodings.pkl"

def encode_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]

            print(f"[INFO] Encoding {filename}...")

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
            else:
                print(f"[WARNING] No faces found in {filename}")

    # Save encodings
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"[INFO] Encodings saved to {ENCODINGS_PATH}")

if __name__ == "__main__":
    encode_faces()
# (yahan poora code paste karo jo maine diya tha)
