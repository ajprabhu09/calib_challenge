import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
def read_files(path):
    videos = glob.glob(f"{path}/*.hevc")
    labels = glob.glob(f"{path}/*.txt")
    return list(zip(videos, labels))

def video_itr(video_file, trace_file):
    cap = cv2.VideoCapture(video_file)

    ret = True
    with open(trace_file,'r+') as tf:
        while ret:
            ret, img = cap.read()
            line = tf.readline()
            line = line[0:-1]
            if line:
                yield (img,np.array([float(x) for x in line.split(" ")]))
    cap.release()

def video_label_itr(path):
    idx = 0
    for (video_file, label_file) in read_files(path):
        for (img, label) in video_itr(video_file, label_file):
            yield img, label, video_file
         

if __name__ == '__main__':
    for i,( image, label) in enumerate(video_itr("./labeled/0.hevc", "./labeled/0.txt")):
        if i == 0:
            prvs = cv.cvtColor(image , cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(image)
            hsv[..., 1] = 255
            continue
        next = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        prvs = next
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == ord('q'):
            break