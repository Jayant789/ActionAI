import cv2
from actionai.predict import (
    predict,
)  # Assuming this is the correct import path for your predict function
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="ActionAI Camera Predict - Real-time action recognition from camera."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the trained model directory"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        help="Window size for processing sequences of keypoints",
    )
    args = parser.parse_args()

    # Load the trained model
    model_dir = args.model
    window_size = args.window_size
    print(f"Loading model from {model_dir} with window size {window_size}")

    # Open a connection to the camera (in this case, camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Perform action recognition on the current frame
            result = predict(model_dir, window_size, frame, save=False, display=True)
            print(result)

            # Display the resulting frame
            cv2.imshow("Action Recognition", frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
