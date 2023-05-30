import cv2
import os
import numpy as np

# Yüz tanıma modelinin yüklenmesi
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yüz tanıma modelinin eğitilmesi
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Yüzlerin kaydedildiği klasörün yolu
face_folder = 'faces/'

# Yüzleri ve etiketlerini saklamak için listeler
faces = []
labels = []
label_mapping = {}  # Etiket-mapping sözlüğü

# Yüzleri ve etiketlerini yükleme
for i, file_name in enumerate(os.listdir(face_folder)):
    if file_name.endswith('.jpg'):
        face_image = cv2.imread(os.path.join(face_folder, file_name), cv2.IMREAD_GRAYSCALE)
        faces.append(face_image)
        label = i
        labels.append(label)
        label_mapping[label] = file_name.split('.')[0]  # .jpg uzantısını kaldırma

# Yüz tanıma modelinin eğitilmesi
face_recognizer.train(faces, np.array(labels))

# Kamera verisi almak için VideoCapture nesnesini başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kamera görüntüsünü al
    ret, frame = cap.read()

    # Gri tonlamaya dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algılama
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan her yüz için işlemler
    for (x, y, w, h) in faces:
        # Kare çerçeve çizme
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Yüzü tanıma
        face_image = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_image)

        # Tanınan yüzleri etiketleme
        if confidence < 100:
            text = f"{label_mapping[label]}"
        else:
            text = "Bilinmeyen Yüz"

        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Sonuçları gösterme
    cv2.imshow('Frame', frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
