import cv2


cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Display the resulting frame
    frame = cv2.flip(frame, 1)  # 1 means horizontal flip
    shape = frame.shape
    print(f"Frame dimensions: {shape[1]}x{shape[0]}")

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()