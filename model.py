import os.path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import filetype
import sys


def validate_file_status(file_path):
    """
    Function to check if file exists, is uncorrupted, and its data type (image or video).
    :param file_path: File path for image or video to be analysed.
    :return: File category of media - e.g. image or video.
    """
    # Check if file exists.
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # Check file type based on headers.
    kind = filetype.guess(file_path)
    if kind is None:
        print("Invalid File: This is neither an image nor a video. Please use a different file.")
        sys.exit(1)

    mime = kind.mime  # e.g. 'image/jpeg' or ' video/mp4'
    is_image = mime.startswith('image')
    is_video = mime.startswith('video')

    if not (is_image or is_video):
        print("Invalid Type: Only images and videos are allowed. Please use a different file.")
        sys.exit(1)

    # Checking file corruption for edge case if header is correct but file corrupted.
    is_corrupted = False

    if is_image:
        img = cv2.imread(file_path)
        if img is None:
            is_corrupted = True
    elif is_video:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened() or not cap.read()[0]:  # Checking if it can open and can read at least first frame.
            is_corrupted = True
        cap.release()

    if is_corrupted:
        print(f"File Corrupted: The {mime.split('/')[0]} file is unreadable. Please use a different file.")
        sys.exit(1)

    # Script valid if this point is reached.
    print(f"Success! {mime} is valid and loaded.")
    file_category = mime.split('/')[0]

    return file_category


def visualise_landmarks(frame, detection_result, eyes=True):
    """
    Draws detected landmarks as points on the image.
    :param frame:
    :param detection_result:
    :param eyes: Set True as default. Visualises eye markers only.
    :return:
    """
    if not detection_result or not detection_result.face_landmarks:
        return frame

    # Get frame dimensions
    height, width, _ = frame.shape

    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]
    left_eye_outline = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_outline = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    eye_markers = left_iris + right_iris + left_eye_outline + right_eye_outline

    # MediaPipe returns a list of pose_landmarks (usually one per person)
    for face_landmarks in detection_result.face_landmarks:
        if eyes:
            for i in eye_markers:
                landmark = face_landmarks[i]
                # Convert normalized coordinates (0-1) to pixel coordinates
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)

                # Draw a small circle at the landmark position
                # cv2.circle(image, center_coordinates, radius, color, thickness)
                cv2.circle(frame, (x_px, y_px), 2, (100, 100, 0), -1)
        else:
            for landmark in face_landmarks:
                # Convert normalized coordinates (0-1) to pixel coordinates
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)

                # Draw a small circle at the landmark position
                # cv2.circle(image, center_coordinates, radius, color, thickness)
                cv2.circle(frame, (x_px, y_px), 2, (100, 100, 0), -1)

    return frame


def mediapipe_initialise(file_path, file_type, eyes=True):
    """
    Function to initialise MediaPipe and visualise result.
    :param file_path: string entry of file path.
    :param file_type: string entry in form "image", "video", or other.
    :param eyes: True to visualise eyes only, set to False if total face to be visualised. True by default.
    :return:
    """
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

        if file_type == 'image':
            # Create a face landmarker instance with the image mode.
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )

            # Landmarker is initialised below.
            with FaceLandmarker.create_from_options(options) as landmarker:
                # Load the input image from an image file.
                mp_image = mp.Image.create_from_file(file_path)
                # Running the task on image mode.
                face_landmarker_result = landmarker.detect(mp_image)

                # Convert MediaPipe image to NumPy array so OpenCV can use it.
                output_image = mp_image.numpy_view()
                # MediaPipe returns it as RGB, OpenCV needs BGR to display colors correctly
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            annotated_image = visualise_landmarks(output_image, face_landmarker_result, eyes)
            cv2.imshow("Annotated image", annotated_image)
            cv2.waitKey(0)

        elif file_type == 'video':
            # Create a face landmarker instance with the video mode.
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO
            )

            # Initialising video and frames per second.
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Landmarker is initialised below.
            with FaceLandmarker.create_from_options(options) as landmarker:
                frame_index = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Calculating Timestamp required for video mode in milliseconds.
                    frame_timestamp_ms = int((frame_index / fps) * 1000)
                    frame_index += 1

                    # Convert BGR (OpenCV) to RGB (MediaPipe).
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Create MediaPipe Image object.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    # Face landmarkers detected for each timestamp.
                    face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                    # Visualise result.
                    annotated_frame = visualise_landmarks(frame, face_landmarker_result, eyes)
                    cv2.imshow('Face Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        else:
            print("ERROR: No valid file.")
            sys.exit(1)

        return "Success"


def eye_landmark_indices():
    """
    Function to output indices of important eye landmarks to assess eye shape and movement.
    :return: 2 lists (left & right eyes) containing lists of indices for iris, eye corners, eyelids, and eye outlines.
    """
    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]

    left_eye_corners = [33, 133]
    right_eye_corners = [362, 263]

    left_eyelids = [159, 145]
    right_eyelids = [386, 374]

    left_eye_outline = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_outline = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    left_eye = [left_iris, left_eye_corners, left_eyelids, left_eye_outline]
    right_eye = [right_iris, right_eye_corners, right_eyelids, right_eye_outline]

    return left_eye, right_eye


def test_run(file_path):
    file_category = validate_file_status(file_path)
    mediapipe_initialise(file_path, file_category, False)


file_path = 'images/test_image.jpeg'
# file_path = 'videos/karentest.mov'

test_run(file_path)
