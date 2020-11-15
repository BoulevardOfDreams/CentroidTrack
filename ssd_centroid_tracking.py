#python .\ssd_centroid_tracking.py -p .\model\MobileNetSSD_deploy.prototxt.txt -m .\model\MobileNetSSD_deploy.caffemodel -v .\detect_material\old_cross.mp4 -l "person"

# import the necessary packages
from imutils.video import FPS
from helper.centroid_track import CentroidTracker
from helper.iou import eliminate_overlap

import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
ap.add_argument("-l", "--label", required=True,
    help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# classes of object that will be detect by the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    
# load model from directory
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#load camera and centroid tracker
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
centroid_tracker = CentroidTracker()

trackers = []
writer = None
labels = []
rect = []

# start the frames per second throughput estimator
fps = FPS().start()
frame_count = 0;

# loop over frames from the video file stream
while True:

    (grabbed, frame) = vs.read()
    
    if frame is None:
        break

    #resize image size to SSD supported size
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    process_rects = []
    
    if frame_count == 0:
    
        # grab the frame dimensions and convert the frame to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        
        #empty tracker
        trackers = []
        
        #empty detected object confidence
        confidences = []
        
        for i in range(0, detections.shape[2]):
            
            if CLASSES[int(detections[0, 0, i, 1])] != args["label"]:
                continue
            
            current_object_confidence_level = detections[0, 0, i, 2]
            
            if current_object_confidence_level > args["confidence"]:
            
                confidences.append(current_object_confidence_level)
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(rgb, rect)
                
                labels.append(args["label"])
                trackers.append(t)
                process_rects.append((startX, startY, endX, endY))
        
        #eliminate overlapping rectangle produce by SSD detection
        process_rects = eliminate_overlap(process_rects,
                                          confidences,
                                          threshold = 0.5)
                                          
        frame_count = 30
    else:
        # loop over each of the trackers
        for t in trackers:
        
            # update the tracker and grab the position of the tracked object
            t.update(rgb)
            pos = t.get_position()
            
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            
            process_rects.append((startX, startY, endX, endY))
            
        frame_count -= 1
        
    #update the tracked object according to newly detected object
    centroid_tracker.update(process_rects)
    
    for id, box in centroid_tracker.rects_stored.items():
        (x, y, end_x, end_y) = box
        (c_x, c_y)           = centroid_tracker.centroid[id]
        
        c_x = int(c_x)
        c_y = int(c_y)
        
        text = "ID {}".format(id)
        
        cv2.rectangle(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, text, (c_x - 10, c_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (c_x, c_y), 4, (0, 255, 0), -1)
        
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()