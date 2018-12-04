# Human-Tracking

Reads the input video and sets the default human detector template for the HOG-SVM detector.
Runs the detector as every 5th video frame is accessed due to bulky nature of the detection algorithm. (significant rise in speed with minimal loss in accuracy since the video doesn't change much in 5 consecutive frames)

In the event of successful detection, the bounding box the envelops the target is returned with its size parameters.
These parameters are then passed onto the KCF tracking algorithm for inititalization.
Since the KCF tracker is a fast algorithm, we dont skip any frames. 
Performs template matching with the initial bounding box to make sure that the human is still within the current bounding box in every iteration.
If the template doesnt match, KCF tracker fails and quits.

All the above frames are written into the output video.


Link to output video : https://drive.google.com/open?id=1ux_lLgEyoe4K-pHfpZLyZL8kzYiCup83
