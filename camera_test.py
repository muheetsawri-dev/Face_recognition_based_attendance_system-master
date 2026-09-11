import cv2

for index in range(5):
    print(f"\nTesting camera index {index}")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()
        print("Frame received:", ret)

        if ret and frame is not None:
            print("Camera WORKS at index:", index)

            cv2.imshow(f"Camera {index}", frame)
            cv2.waitKey(3000)

        cap.release()

cv2.destroyAllWindows()