import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()

amount_image_extract = 20
amount_frame_pass = 20
count_img = 0
frame = 0

video_path = os.listdir("Video")[0]
cap = cv2.VideoCapture(f"Video/{video_path}")

while True:
    frame += 1
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not frame % amount_frame_pass == 0:
        continue
    faces = detector(gray)
    if len(faces) > 0:
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            img_crop = img[y1:y2, x1:x2]
            count_img += 1
            cv2.imwrite(f"DataImage/Chu nha/Face_{count_img}.jpg", img_crop)
            print(f"Saving img {count_img}")
    if count_img >= amount_image_extract:
        cap.release()
        break