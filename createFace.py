import cv2
import os

# Yüz tanıma modelinin yüklenmesi
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera verisi almak için VideoCapture nesnesini başlatma
cap = cv2.VideoCapture(0)

# Yüzlerin saklanacağı klasörün oluşturulması
face_folder = 'faces/'
if not os.path.exists(face_folder):
    os.makedirs(face_folder)

# Yüzleri kaydetmek için kullanıcıdan kimlik (etiket) girişi
label = input("Yüz etiketi girin: ")

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
        
        # Yüzü saklama
        face_image = gray[y:y+h, x:x+w]
        cv2.imwrite(face_folder + label + '.jpg', face_image)
    
    # Sonuçları gösterme
    cv2.imshow('Frame', frame)
    
    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
