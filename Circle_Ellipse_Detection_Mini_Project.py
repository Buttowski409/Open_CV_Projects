import cv2
import numpy as np

image = cv2.imread('blobs22.jpg', 0)

detector = cv2.SimpleBlobDetector_create()
key_points = detector.detect(image)
blank = np.zeros((1, 1))

blobs = cv2.drawKeypoints(image, key_points, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

no_of_blobs = len(key_points)
text = "no. of Blobs: " + str(no_of_blobs)
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
cv2.imshow('Blob Detect', blobs)
cv2.waitKey(0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

params.filterByCircularity = True
params.minCircularity = 0.85

params.filterByConvexity = False
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keyPoints = detector.detect(image)
blank = np.zeros((1, 1))

blobs = cv2.drawKeypoints(image, keyPoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
no_of_blobs = len(keyPoints)
text = "no. of Blobs: " + str(no_of_blobs)
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
cv2.imshow('Circle Detect', blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
