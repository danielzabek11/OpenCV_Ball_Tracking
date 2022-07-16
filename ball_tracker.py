from collections import deque
import numpy as np
import cv2
import imutils


# HSV color ranges for object of interest
greenLower = (30, 39, 47)
greenUpper = (55, 178, 147)

# Setting the length of line showing previous object movment
pts = deque(maxlen=64)

# Accessing webcam
cap = cv2.VideoCapture(0)

while True:
	check, frame = cap.read()

	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
	# Erosion and dialation used for better object tracking accuracy
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# Find external contours of object
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	center = None
	# Check if any contour was found
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		#centroid of object using image moment
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 15:
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# Append centroid to list of pts
	pts.appendleft(center)

    	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)

	cv2.imshow("Frame", frame)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()