import cv2
from util import get_parking_spots_bboxes
from networkx import connected_components

mask_path = './data/mask_crop.png'
video_path = './data/parking_crop.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

ret = True
while ret:
    ret, frame = cap.read()

    for spot in spots:
        x1, y1, w, h = spot

        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0))

    if frame is not None:
        cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()