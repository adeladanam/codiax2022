import cv2

# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
    # Capture the video frame
    ret, frame = vid.read()
    if frame is None:
        continue
    # PROCESS FRAME
    cv2.imshow('frame', frame)
    # Quit using q button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()