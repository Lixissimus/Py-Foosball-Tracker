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

  kf = cv2.KalmanFilter(stateSize, measSize, contrSize, cv2.CV_32F)

  state = np.zeros((stateSize, 1), np.float32)
  meas = np.zeros((measSize, 1), np.float32)

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

  kf.measurementNoiseCov = np.identity(measSize, np.float32) * 1e-1

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

    cv2.imshow('Frame', frame)

    res = np.copy(frame)

    if found:
      kf.transitionMatrix.put(2, dT)
      kf.transitionMatrix.put(9, dT)

      state = kf.predict()

      x = state.item(0)
      y = state.item(1)
      w = state.item(4)
      h = state.item(5)
      topLeft = (int(x - w/2) - 3, int(y - h/2) - 3)
      bottomRight = (int(x + w/2) + 3, int(y + h/2) + 3)

      res = cv2.rectangle(res, topLeft, bottomRight, (255, 0, 0), 2)

    # blur image
    blur = cv2.GaussianBlur(frame, (5, 5), 3.)
    # convert to HSV
    frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # color thresholding
    lowerColor = np.array([MIN_H_BLUE / 2, 100, 80])
    upperColor = np.array([MAX_H_BLUE / 2, 255, 255])
    rangeRes = cv2.inRange(frmHsv, lowerColor, upperColor)

    # morphological opening
    kernel = np.ones((5,5), np.uint8)
    rangeRes = cv2.morphologyEx(rangeRes, cv2.MORPH_OPEN, kernel)

    cv2.imshow("Binary", rangeRes)

    # find objects
    _, contours, _ = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find largest area
    maxArea = -1
    maxIdx = -1
    for idx, cnt in enumerate(contours):
      area = cv2.contourArea(cnt)
      if area > 500 and area > maxArea:
        maxArea = area
        maxIdx = idx

    if maxIdx >= 0:
      # a ball was detected
      notFoundCount = 0

      # draw bounding rect with center point
      x,y,w,h = cv2.boundingRect(contours[maxIdx])
      cv2.rectangle(res, (x,y), (x+w,y+h), (0,255,0), 2)
      center = (x + w/2, y + h/2)
      cv2.circle(res, center, 3, (0,255,0), -1)

      # update measurements (z_x, z_y, z_w, z_h)
      meas.put(0, center[0])
      meas.put(1, center[1])
      meas.put(2, w)
      meas.put(3, h)

      if not found:
        # first detection, do initialization
        kf.errorCovPre.put(0, 1)
        kf.errorCovPre.put(7, 1)
        kf.errorCovPre.put(14, 1)
        kf.errorCovPre.put(21, 1)
        kf.errorCovPre.put(28, 1)
        kf.errorCovPre.put(35, 1)

        state.put(0, meas.item(0))
        state.put(1, meas.item(1))
        state.put(2, 0)
        state.put(3, 0)
        state.put(4, meas.item(2))
        state.put(5, meas.item(3))

        found = True
      else:
        # correct kalman filter with measurements
        # pdb.set_trace()
        kf.correct(meas)
    else:
      # no ball was detected
      notFoundCount += 1
      if notFoundCount >= 50:
        found = False
      else:
        kf.statePost = state

    cv2.imshow("Contours", res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()