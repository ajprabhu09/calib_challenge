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
    for i, (frame, label) in enumerate(video_itr("./labeled/0.hevc", "./labeled/0.txt")):
        if i == 0:
            # params for ShiTomasi corner detection
            feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )
            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15, 15),
                            maxLevel = 2,
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
            # Create some random colors
            color = np.random.randint(0, 255, (100, 3))
            # Take first frame and find corners in it
            old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame)

            p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            continue
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        k = cv.waitKey(30) & 0xff
        if k == ord('q'):
            break