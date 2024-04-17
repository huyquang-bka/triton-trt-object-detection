import cv2


def main(index):
    # init grpc

    # Load the image
    path = "rtsp://192.168.1.198:8554/camera/tail"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(path)
            continue
        cv2.imshow("frame", frame)
        cv2.imwrite("frame.jpg", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main(0)
