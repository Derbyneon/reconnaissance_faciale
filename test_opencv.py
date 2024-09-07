#dectection de visage sans deep learning
"""import cv2
import os

image = cv2.imread("jacob.jpg")
print("Dimensions de l'image: ",image.shape)

# on convertit l'image en noir et blanc
# l'algorithme que nous allons utilisé a besoin de ce pretraitement
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# on a besoin de ce fichier on ne fait ici
# que la prédiction, pas le training 
#https://github.com/opencv/opencv/tree/master/data/haarcascades

# on charge notre modèle
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# on verifie que le modèle a bien été chargée
if face_cascade.empty()==True:
	print("Le fichier n'est pas chargé: ", face_cascade.empty())
else:
	print("Le fichier est chargé.")

# On cherche tous les visages disponibles dans l'image
faces = face_cascade.detectMultiScale(image_gray, 1.1, 5)
# on écrit dans la console le nombre de visages que  l'algorithme a détecté
print(f"{len(faces)} visages detectés dans l'image.")

# on dessine un rectangle autour de chaque visage
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

print ("C'est tout bon!")

# on sauvegarde l'image
cv2.imwrite("new.jpg", image)"""




#dectection de visage avec deep learning
"""
import cv2
import numpy as np

# Chemins vers les fichiers du modèle pré-entraîné
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Charger une image
image = cv2.imread('jacob.jpg')

# Préparer l'image : convertir en blob pour le passer au modèle de détection
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))

# Passer le blob au réseau de neurones et obtenir les détections
net.setInput(blob)
detections = net.forward()

# Boucle sur les détections
for i in range(0, detections.shape[2]):
    # Extraire la confiance (c'est-à-dire la probabilité) associée à la prédiction
    confidence = detections[0, 0, i, 2]

    # Filtrer les détections faibles en s'assurant que la confiance est supérieure à un seuil
    if confidence > 0.5:
        # Calculer les coordonnées (x, y) de la boîte englobante pour l'objet
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Dessiner la boîte englobante autour du visage avec la probabilité
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Afficher l'image de sortie
cv2.imshow("Detecte moi un visage", image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""




#avec la webcam 
import cv2
import numpy as np

# Chemins vers les fichiers du modèle pré-entraîné
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Démarrer le flux vidéo de la caméra
cap = cv2.VideoCapture(0)

while True:
    # Lire une frame de la vidéo
    ret, frame = cap.read()
    
    # S'assurer que la frame a été lue correctement
    if not ret:
        break

    # Convertir la frame en blob pour le passer au modèle de détection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # Passer le blob au réseau et obtenir les détections
    net.setInput(blob)
    detections = net.forward()

    # Boucle sur les détections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Dessiner la boîte englobante avec la probabilité
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Afficher la frame avec les détections
    cv2.imshow("Frame", frame)
    
    # Sortir de la boucle si on appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer le flux vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()