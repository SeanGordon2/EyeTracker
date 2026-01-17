import os.path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

image = cv2.imread('images/pathology_test.jpg')
# image = cv2.imread('images/seantest.jpg')  # 'images/test_image.jpeg'
# image = cv2.imread('images/eyes_only.jpg')
height, width, _ = image.shape


# Path to downloaded .task file.
model_path = os.path.abspath("face_landmarker.task")

# Ensure .task file appropriately saved.
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
else:
    # Configure options for Eye + Iris tracking for Image.
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Landmarker is initialised at this point.

        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file('images/pathology_test.jpg')
        # mp_image = mp.Image.create_from_file('images/seantest.jpg')  # 'images/test_image.jpeg'

        # Running the task.
        face_landmarker_result = landmarker.detect(mp_image)

    for facial_landmarks in face_landmarker_result.face_landmarks:
        # print(len(facial_landmarks))
        # print(facial_landmarks[0].x, facial_landmarks[0].y, facial_landmarks[0].z)
        landmark_point_list = []
        for i in range(0, 478):
            point = facial_landmarks[i]
            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(image, (x, y), 2, (100, 100, 0), -1)  # To place coloured circle by points.
            # cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))  # To place numbers by text.

    cv2.imshow("image", image)
    cv2.waitKey(0)




if __name__ == '__main__':
    print('Complete!')
