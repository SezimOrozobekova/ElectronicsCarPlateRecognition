import cv2
import pytesseract

# Путь к файлу каскада Хаара для обнаружения номерных знаков
harcascade = "haarcascade_russian_plate_number.xml"
image_path = "images/image (7).jpg"  # Путь к вашему изображению

min_area = 500

# Путь к установленному Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Загрузка изображения
frame = cv2.imread(image_path)

# Инициализация детектора номерных знаков
plate_detector = cv2.CascadeClassifier(harcascade)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Обнаружение номерных знаков
plates = plate_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)


for (x, y, w, h) in plates:
    area = w * h
    if area > min_area:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_img = frame[y:y + h, x:x + w]

        gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Применение предварительной обработки к изображению
        gray_plate_img = cv2.resize(gray_plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray_plate_img = cv2.GaussianBlur(gray_plate_img, (5, 5), 0)

        # Распознавание текста с помощью Tesseract с настройками параметров
        plate_text = pytesseract.image_to_string(gray_plate_img, config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

        roi_resized = cv2.resize(plate_img, (400, 100))
        cv2.imshow("ROI", roi_resized)
        frame = cv2.resize(frame, (700, 512))
        cv2.putText(frame, "Extracted Text: " + plate_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                    2)
        print(plate_text)

# Показать результаты
cv2.imshow("PythonGeeks", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
