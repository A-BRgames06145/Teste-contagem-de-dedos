import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

desenho_mp = mp.solutions.drawing_utils
maos_mp = mp.solutions.hands

dedos = [4, 8, 12, 16, 20]

maos = maos_mp.Hands(min_detection_confidence= 0.8, min_tracking_confidence = 0.5)

def desenharMarcasMao(image, marcasMao):
    if marcasMao:
        for marcaMao in marcasMao:
            desenho_mp.draw_landmarks(image, marcaMao, maos_mp.HAND_CONNECTIONS)

while True:
    success, image = webcam.read()

    image = cv2.flip(image, 1)
    result = maos.process(image)
    
    marcasMao = result.multi_hand_landmarks
    desenharMarcasMao(image, marcasMao)

    cv2.imshow('Webcam', image)

    key = cv2.waitKey(1)

    if (key == 32):
        break

cv2.destroyAllWindows()