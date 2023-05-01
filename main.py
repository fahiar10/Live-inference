# This is a sample Python script.
import cv2,os
from ultralytics import YOLO


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = YOLO('best.pt')
    video_path = os.path.join(os.getcwd(),"beevideo.mp4")
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
