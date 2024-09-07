import dlib
import face_recognition
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Capture vidéo
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

# Charger les images d'exemple
nicolas_image = face_recognition.load_image_file("jacob.jpg")
nicolas_face_encoding = face_recognition.face_encodings(nicolas_image)[0]

maureen_image = face_recognition.load_image_file("joachim.jpg")
maureen_face_encoding = face_recognition.face_encodings(maureen_image)[0]

# Encodages et noms de visages connus
known_face_encodings = [nicolas_face_encoding, maureen_face_encoding]
known_face_names = ["Jacob", "Joachim"]

# Variables initiales
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Fonction de traitement des frames (pour exécution asynchrone)
def process_frame(rgb_small_frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    return face_locations, face_names

# Utiliser un ThreadPoolExecutor pour exécuter la détection faciale en parallèle
with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        ret, frame = video_capture.read()
        
        # Redimensionner la frame pour accélérer le traitement
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Traiter une frame sur deux
        if process_this_frame:
            future = executor.submit(process_frame, rgb_small_frame, known_face_encodings, known_face_names)
            face_locations, face_names = future.result()
        
        process_this_frame = not process_this_frame
        
        # Afficher les résultats
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Afficher l'image résultante
        cv2.imshow('Video', frame)
        
        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer la webcam et fermer les fenêtres
video_capture.release()
cv2.destroyAllWindows()
