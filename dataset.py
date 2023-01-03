import cv2
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
            yield img, label
         


# ransac_iterations,ransac_threshold,ransac_ratio = 350,13,0.93
# print(RANSAC(process_lines()['x'],ransac_iterations,ransac_threshold,ransac_ratio))

if __name__ == '__main__':
    for img, label in video_label_itr("labeled"):
        print(label)
