import cv2
import face_recognition
import os
import numpy as np

# Chemin vers le dossier contenant les images de visages connus
KNOWN_FACES_DIR = 'known_faces'

# Charger les visages connus et leurs noms
known_face_encodings = []
known_face_names = []

for image_name in os.listdir(KNOWN_FACES_DIR):
    # Charger chaque image avec OpenCV
    image_path = os.path.join(KNOWN_FACES_DIR, image_name)
    image_bgr = cv2.imread(image_path)

    # Vérifier si l'image a été chargée correctement
    if image_bgr is None:
        print(f"Erreur de chargement de l'image: {image_name}")
        continue

    # Vérifier que l'image est bien en 8 bits par canal
    if image_bgr.dtype != np.uint8:
        print(f"L'image {image_name} n'est pas au format 8 bits.")
        continue

    # Convertir l'image de BGR (OpenCV) en RGB (face_recognition)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Obtenir l'encodage du visage
    encodings = face_recognition.face_encodings(image_rgb)
    if encodings:  # S'assurer qu'un encodage a été trouvé
        encoding = encodings[0]
        known_face_encodings.append(encoding)
        # Utiliser le nom du fichier (sans l'extension) comme nom de la personne
        name = os.path.splitext(image_name)[0]
        known_face_names.append(name)
    else:
        print(f"Aucun encodage trouvé pour l'image: {image_name}")

# Démarrer le flux vidéo de la caméra
cap = cv2.VideoCapture(0)

while True:
    # Lire une frame de la vidéo
    ret, frame = cap.read()
    
    # S'assurer que la frame a été lue correctement
    if not ret:
        break

    # Convertir l'image BGR (OpenCV) en RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Trouver toutes les localisations et encodages des visages dans la frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Parcourir chaque visage détecté dans la frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparer le visage détecté aux visages connus
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Inconnu"

        # Si une correspondance est trouvée
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Dessiner une boîte autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Afficher le nom de la personne
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Afficher la frame avec les détections
    cv2.imshow("Reconnaissance Faciale", frame)
    
    # Sortir de la boucle si on appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer le flux vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
