import cv2
import os
import uuid


def capture_face_data(user_id, num_images=30, output_dir="dataset"):
    if not user_id:
        raise ValueError("User ID cannot be empty.")

    user_path = os.path.join(output_dir, user_id)
    os.makedirs(user_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot access the webcam.")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"Starting capture for user {user_id}. Look at the camera...")

    count = 0
    while count < num_images:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            img_filename = os.path.join(user_path, f"user.{user_id}.{uuid.uuid4()}.jpg")
            cv2.imwrite(img_filename, gray[y:y + h, x:x + w])
            cv2.imshow('Face Capture', img)

        if cv2.waitKey(100) & 0xFF == 27:  # ESC to exit
            break

    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        raise RuntimeError("No faces were detected. Please try again.")

    print(f"Successfully captured {count} images for user {user_id}.")


if __name__ == "__main__":
    user_id = input("Enter the user ID for capturing face data: ").strip()
    capture_face_data(user_id)
