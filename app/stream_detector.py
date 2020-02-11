from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
sys.path.append('../')
from people_counter import Detector

class StreamDetector:

    def __init__(self, stream_url):
        self.stream_url = stream_url

    def get_inference(self):
        stream = cv2.VideoCapture(self.stream_url)
        detector = Detector(20,0.4)
        frame_counter = 0

        while stream.isOpened():
            valid_frame , frame = stream.read()
            if(valid_frame):
                frame_counter += 1
                frame = detector.get_predicted_frame(frame, frame_counter)
                frame_enconde = cv2.imencode('.jpg', frame)[1]
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n'+frame_enconde.tostring()+b'\r\n')

        stream.release()
        del stream

        

