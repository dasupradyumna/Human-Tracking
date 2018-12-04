#run on python3.5
#although the video is written at 25fps, the actual processing takes a couple of minutes

import sys
import cv2
import numpy as np
import imutils

def sliding_window(img, step, wSize):		# produces sliding windows on an input image
	for y in range(0, img.shape[0] - wSize[1] + 1, step):
		for x in range(0, img.shape[1] - wSize[0] + 1, step):
			yield (x, y, img[y:y + wSize[1], x:x + wSize[0]])

def HEQ(img):			#histogram equalization on the input image 
	img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	img = cv2.GaussianBlur(img,(3,3),0)
	return img 

def matching(img,box,targ):			#checks if a target template is present in the input image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	targ = cv2.cvtColor(targ,cv2.COLOR_BGR2GRAY)
	x,y,w,h = box

	res = cv2.matchTemplate(img[int(y):int(y+h),int(x):int(x+w)],targ,cv2.TM_SQDIFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	threshold = 0.3
	loc = np.where(min_val<=threshold)
	if len(loc[0]) == 0 :
		return False
	else :
		return True

def run_main():
	cap = cv2.VideoCapture('terrorist.mp4') #1920x1080 29fps 458 frames
	out = cv2.VideoWriter('detect.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1920,1080))

	if cap.isOpened() == False :
		print('Error : Video file missing')
		sys.exit(1)

	hog = cv2.HOGDescriptor()					#initializing the HOG SVM detector
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	tracker = cv2.Tracker_create('KCF')		#for opencv version <= 3.2
	#tracker = cv2.TrackerKCF_create()		#for opencv version >= 3.3

	c = 0
	detected = 0
	while True :
		ret,frame = cap.read()
		if ret == False :
			break
		frame = imutils.resize(frame, width = 2*frame.shape[1])		#upscaling to better detect HOG features

		if detected == 1 :			#tracking starts after detection is successful
			ok, bbox = tracker.update(HEQ(frame))
			x,y,w,h = bbox

			if matching(HEQ(frame),bbox,temp) :		#tracking only continues if there is human still in tracking window
				topL = (int(x),int(y))
				bottomR = (int(x+w),int(y+h))
				cv2.rectangle(frame,topL,bottomR,(0,0,255),5)

		if c%5 == 0 and detected == 0 :				#using one every 5 frames for the heavy detection alogrithm
			for (X,Y,img) in sliding_window(HEQ(frame),160,(280,280)):			#histogram equalizing before detection to reduce haze and improce contrast
				(rects,weights) = hog.detectMultiScale(img, winStride = (6,6), padding = (4,4), scale = 1.5)
				for (x,y,w,h) in rects :
					cv2.rectangle(frame, (X+x,Y+y), (X+x+w,Y+y+h), (0,0,255), 5)
				
				if len(rects) != 0 :		#confirming first detection and exiting to tracking algorithm
					detected = 1
					x,y,w,h = np.array(rects[0]) + np.array([X,Y,0,0])
					bbox = (x,y,w,h)

					temp = frame[int(y):int(y+h),int(x):int(x+w)]	#template for matching later tracking windows
					tracker.init(HEQ(frame),bbox)
					break

		frame = imutils.resize(frame, width = 1920)
		out.write(frame)	#writing frames to video
		
		c += 1

	cap.release()		#releasing all video objects
	out.release()

if __name__ == '__main__':
 	run_main() 