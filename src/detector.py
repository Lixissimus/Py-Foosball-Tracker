import numpy as np
import cv2

import pdb

class KalmanBallDetector:
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
        self.tablePath = np.array([[275,53], [570,53], [696,423], [154,428]])
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

class ParticleBallDetector:
    def __init__(self):
        self.likelihoodField = None
        self.tablePath = np.array([[275,53], [570,53], [696,423], [154,428]])
        self.tableMask = None

        self.deskewWidth = 370
        self.deskewHeight = 500

        self.nrParticles = 500
        self.particles = None
        self.weights = None

        self.motionNoiseStdDev = 50

    def createLikelihoodField(self, frame):
        if self.tableMask is None:
            self.createTableMask(frame)

        self.likelihoodField = np.zeros(
            (frame.shape[0], frame.shape[1]), np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # detect bars by finding lines
        minLineLength = 190
        lines = cv2.HoughLines(edges, 1, np.pi/180, minLineLength)
        if lines is not None:
            for line in lines:
                rho,theta = line[0]
                thetaDeg = theta/np.pi*180

                # we're looking for horizontal lines,
                # filter by line angle
                if thetaDeg < 88 or thetaDeg > 92:
                    continue

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(self.likelihoodField,(x1,y1),(x2,y2),(255,255,255),2)

        
        # self.likelihoodField = cv2.bitwise_and(
        #     self.likelihoodField, self.likelihoodField, mask=self.tableMask)

        # deskew likelihood field
        pts1 = np.float32(self.tablePath)
        pts2 = np.float32(
            [[0,0], [self.deskewWidth,0], 
            [self.deskewWidth,self.deskewHeight], [0,self.deskewHeight]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        self.likelihoodField = cv2.warpPerspective(
            self.likelihoodField, M, (self.deskewWidth,self.deskewHeight))

        # apply some gaussian noise
        self.likelihoodField = cv2.GaussianBlur(
            self.likelihoodField, (21,21), 10)

        cv2.imshow('likelihood field', self.likelihoodField)

    def createTableMask(self, frame):
        self.tableMask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.fillPoly(self.tableMask, [self.tablePath], (255,255,255))

    def initParticles(self):
        rand = np.random.uniform(0, 1, (self.nrParticles,2))
        self.particles = np.full(
            (self.nrParticles,2), (self.deskewWidth, self.deskewHeight)) * rand
        self.weights = np.full(self.nrParticles, 1./self.nrParticles)

    def applyMotion(self):
        noise = np.random.normal(
            0, self.motionNoiseStdDev, (self.nrParticles,2))
        self.particles += noise

    def calculateWeights(self):
        weightSum = 0
        for idx, particle in enumerate(self.particles):
            x, y = particle[0], particle[1]
            if x < 0 or x >= self.deskewWidth or y < 0 or y >= self.deskewHeight:
                self.weights[idx] = 0
            else:
                self.weights[idx] = self.likelihoodField[y][x]/255.0
                weightSum += self.weights[idx]
        for idx in range(0, self.nrParticles):
            self.weights[idx] /= weightSum

    def resample(self):
        weightSum = 1.0
        stepSize = weightSum / self.nrParticles
        offset = np.random.uniform(0, stepSize)
        result = []

        idx = 0
        passed = self.weights[idx]

        while (offset <= weightSum):
            while passed < offset:
                idx += 1
                passed += self.weights[idx]
            while (offset <= passed):
                result.append(idx)
                offset += stepSize

        newParticles = np.zeros((self.nrParticles, 2))
        for idx, particleIdx in enumerate(result):
            newParticles[idx][0] = self.particles[particleIdx][0]
            newParticles[idx][1] = self.particles[particleIdx][1]

        self.particles = newParticles


    def drawParticles(self, frame):
        locations = self.particles.astype(int)
        for loc in locations:
            cv2.circle(frame, (loc[0],loc[1]), 1, (0,255,0), -1)

    def initFilter(self, frame):
        self.createLikelihoodField(frame)
        self.initParticles()

    def feedFrame(self, frame):
        # for some debug printing
        def onMouse(evt, x, y, a, b):
            print x, y

        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', onMouse)
        cv2.imshow('frame', frame)

        # deskew frame
        pts1 = np.float32(self.tablePath)
        pts2 = np.float32(
            [[0,0], [self.deskewWidth,0], 
            [self.deskewWidth,self.deskewHeight], [0,self.deskewHeight]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        deskewed = cv2.warpPerspective(
            frame, M, (self.deskewWidth,self.deskewHeight))

        self.applyMotion()
        self.calculateWeights()
        self.resample()
        self.drawParticles(deskewed)

        cv2.imshow('deskewed', deskewed)


