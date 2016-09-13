import numpy as np
import cv2
import sys

from detector import KalmanBallDetector, ParticleBallDetector

if __name__ == "__main__":
    print "Starting Foosball Tracker"
    print "Using OpenCV Version", cv2.__version__

    cap = cv2.VideoCapture("C:\\Users\\Felix\\Documents\\Projects\\Py-Foosball-Tracker\\assets\\videos\\test-vid-blue-yellow-better.avi")
    if not cap.isOpened():
        print "Could not access camera!"
        sys.exit(1)

    # ballDetector = KalmanBallDetector()
    ballDetector = ParticleBallDetector()
    ret, frame = cap.read()
    ballDetector.initFilter(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        ballDetector.feedFrame(frame)
        # x, y, _, _, w, h = ballDetector.getBestEstimate() 

        # cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), (255,0,0), 2)
        
        # cv2.imshow('frame', frame)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()