import numpy as np
import cv2
import sys

import pdb

if __name__ == "__main__":
  print "Starting Foosball Tracker"
  print "Using OpenCV Version", cv2.__version__

  MIN_H_BLUE = 200
  MAX_H_BLUE = 300
  
  stateSize = 6
  measSize = 4
  contrSize = 0
  valType = cv2.CV_32F

  kf = cv2.KalmanFilter(stateSize, measSize, contrSize, valType)

  state = np.zeros((stateSize, 1), np.float32)
  meas = np.array((measSize, 1), np.float32)

  kf.transitionMatrix = np.identity(stateSize, np.float32)

  kf.measurementMatrix = np.zeros((measSize, stateSize), np.float32)
  kf.measurementMatrix.put(0, 1)
  kf.measurementMatrix.put(7, 1)
  kf.measurementMatrix.put(16, 1)
  kf.measurementMatrix.put(23, 1)

  kf.processNoiseCov = np.zeros((stateSize, stateSize), np.float32)
  kf.processNoiseCov.put(0, 1e-2)
  kf.processNoiseCov.put(7, 1e-2)
  kf.processNoiseCov.put(14, 2)
  kf.processNoiseCov.put(21, 1)
  kf.processNoiseCov.put(28, 1e-2)
  kf.processNoiseCov.put(35, 1e-2)

  kf.measurementNoiseCov = np.identity(stateSize, np.float32) * 1e-1

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print "Could not access camera!"
    sys.exit(1);

  ticks = 0.
  found = False
  notFoundCount = 0


  while True:
    precTicks = ticks
    ticks = cv2.getTickCount()
    dT = (ticks - precTicks) / cv2.getTickFrequency()

    (ret, frame) = cap.read()
    if not ret:
      break

    cv2.imshow('frame', frame)

    res = np.copy(frame)

    if found:
      kf.transitionMatrix.put(2, dT)
      kf.transitionMatrix.put(9, dT)
      print "dT:", dT

      state = kf.predict()

      x = state.item(0)
      y = state.item(1)
      w = state.item(4)
      h = state.item(5)
      topLeft = (x - w/2, y - h/2)
      bottomRight = (x + w/2, y + h/2)

      res = cv2.rectangle(res, topLeft, bottomRight, (255, 0, 0))


    blur = cv2.GaussianBlur(frame, (5, 5), 3.)
    frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # rangeRes = np.zeros(frame.shape, np.uint8)
    lowerColor = np.array([MIN_H_BLUE / 2, 100, 80])
    upperColor = np.array([MAX_H_BLUE / 2, 255, 255])
    rangeRes = cv2.inRange(frmHsv, lowerColor, upperColor)


    # TODO: erode, dilate

    contours = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pdb.set_trace()

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()