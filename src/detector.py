import numpy as np
import cv2

class BallDetector:
    def __init__(self):
        # detection color setup
        self.hLow = 8
        self.hHigh = 42
        self.sLow = 106
        self.sHigh = 255
        self.vLow = 102
        self.vHigh = 255

        self.lowerColor = np.array([self.hLow, self.sLow, self.vLow])
        self.upperColor = np.array([self.hHigh, self.sHigh, self.vHigh])

        # kalman filter setup
        self.stateSize = 6
        self.measurementSize = 4
        self.controlSize = 0

        self.kalmanFilter = cv2.KalmanFilter(
            self.stateSize,
            self.measurementSize,
            self.controlSize,
            cv2.CV_32F)

        # X = [x, y, v_x, v_y, w, h]
        self.state = np.zeros((self.stateSize,1), np.float32)
        self.bestEstimate = (-1,-1,0,0,-1,-1)
        # Z = [z_x, z_y, z_w, z_h]
        self.measurement = np.zeros((self.measurementSize,1), np.float32)

        #     [1  0  dT 0  0  0]
        #     [0  1  0  dT 0  0]
        # F = [0  0  1  0  0  0]
        #     [0  0  0  1  0  0]
        #     [0  0  0  0  1  0]
        #     [0  0  0  0  0  1]
        self.kalmanFilter.transitionMatrix = np.identity(
            self.stateSize, np.float32)

        # H
        self.kalmanFilter.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)

        # covariance of X
        # values tuned by trial and error
        val = 1e-1
        self.kalmanFilter.processNoiseCov = np.array([
            [val, 0, 0, 0, 0, 0],
            [0, val, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, val, 0],
            [0, 0, 0, 0, 0, val]], dtype=np.float32)

        # covariance of Z
        # values tuned by trial and error
        self.kalmanFilter.measurementNoiseCov = np.identity(
            self.measurementSize, np.float32) * 1e-1

        # general tracking setup
        self.tracking = False
        self.ticks = 0
        self.lostBallCounter = 0
        self.previousFrame = None
        self.tablePath = np.array([[386,50], [679,84], [631,478], [53,323]])
        # region-of-interest radius
        self.roiRadius = 100
        self.lostBallThreshold = 50

    def feedFrame(self, frame):
        prevTicks = self.ticks
        self.ticks = cv2.getTickCount()
        deltaT = (self.ticks - prevTicks) / cv2.getTickFrequency()

        if self.tracking:
            self.kalmanFilter.transitionMatrix.put(2, deltaT)
            self.kalmanFilter.transitionMatrix.put(9, deltaT)

            self.state = self.kalmanFilter.predict()
            self.bestEstimate = (
                int(self.state.item(0)),
                int(self.state.item(1)),
                int(self.state.item(2)),
                int(self.state.item(3)),
                int(self.state.item(4)),
                int(self.state.item(5)))
            dbg = np.copy(frame)
            x = self.state.item(0)
            y = self.state.item(1)
            w = self.state.item(4)
            h = self.state.item(5)
            topLeft = (int(x - w/2) - 3, int(y - h/2) - 3)
            bottomRight = (int(x + w/2) + 3, int(y + h/2) + 3)

            cv2.rectangle(dbg, topLeft, bottomRight, (255, 0, 0), 2)


        tableMask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.fillPoly(tableMask, [self.tablePath], (255,255,255))
        maskedFrame = cv2.bitwise_and(frame, frame, mask=tableMask)

        if self.previousFrame == None:
            self.previousFrame = maskedFrame
            return -1, -1

        maskedGray1 = cv2.cvtColor(self.previousFrame, cv2.COLOR_BGR2GRAY)
        maskedGray2 = cv2.cvtColor(maskedFrame, cv2.COLOR_BGR2GRAY)

        self.previousFrame = maskedFrame

        # create a diff image and threshold it
        diff = cv2.absdiff(maskedGray1, maskedGray2)
        _, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        # dilate it to increase the regions of interest
        dilateKernel = np.ones((5,5), np.uint8)
        diff = cv2.dilate(diff, dilateKernel)
        # add region of interest around predicted ball position
        cv2.circle(
            diff, 
            (int(self.state.item(0)),int(self.state.item(1))),
            self.roiRadius,
            (255,255,255),
            -1)
        # apply region-of-interest mask
        maskedFrame = cv2.bitwise_and(maskedFrame, maskedFrame, mask=diff)
        cv2.imshow('kalman', maskedFrame)

        blurredFrame = cv2.GaussianBlur(maskedFrame, (5,5), 3.)
        # convert to HSV for color range thresholding
        frameHsv = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

        # color thresholding
        thresholdFrame = cv2.inRange(frameHsv, self.lowerColor, self.upperColor)

        # morphological opening
        kernel = np.ones((2,2), np.uint8)
        thresholdFrame = cv2.morphologyEx(thresholdFrame,cv2.MORPH_OPEN, kernel)

        # find largest object
        _, contours, _ = cv2.findContours(
            thresholdFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        maxArea = -1
        maxIdx = -1
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 20 and area > maxArea:
                maxArea = area
                maxIdx = idx

        if maxIdx >= 0:
            # a ball was detected
            self.lostBallCounter = 0

            # update measurement
            x,y,w,h = cv2.boundingRect(contours[maxIdx])
            self.measurement.put(0, x+w/2)
            self.measurement.put(1, y+h/2)
            self.measurement.put(2, w)
            self.measurement.put(3, h)

            cv2.rectangle(maskedFrame, (x-w/2,y-h/2), (x+w/2,y+h/2), (0,255,0), 2)
            cv2.imshow('dbg', maskedFrame)

            if not self.tracking:
                # first detection after ball was lost completely
                self.kalmanFilter.errorCovPre.put(0, 1)
                self.kalmanFilter.errorCovPre.put(7, 1)
                self.kalmanFilter.errorCovPre.put(14, 1)
                self.kalmanFilter.errorCovPre.put(21, 1)
                self.kalmanFilter.errorCovPre.put(28, 1)
                self.kalmanFilter.errorCovPre.put(35, 1)

                self.state.put(0, self.measurement.item(0))
                self.state.put(1, self.measurement.item(1))
                self.state.put(2, 0)
                self.state.put(3, 0)
                self.state.put(4, self.measurement.item(2))
                self.state.put(5, self.measurement.item(3))

                self.tracking = True
            else:
                # correct kalman filter with measurement
                self.kalmanFilter.correct(self.measurement)
        else:
            # no ball was detected
            self.lostBallCounter += 1
            if self.lostBallCounter > self.lostBallThreshold:
                self.tracking = False
            else:
                self.kalmanFilter.statePost = self.state


    def getBestEstimate(self):
        return self.bestEstimate
        # return int(self.state.item(0)), int(self.state.item(1))
