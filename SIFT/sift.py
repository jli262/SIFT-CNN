import numpy as np
import cv2
sift = cv2.xfeatures2d.SIFT_create()
for i in range(1, 21):
    if(i<10):
        querypath = r"D:\SIFT-CNN\CNN\Queries\0" + str(i) + ".jpg"
    else:
        querypath = r"D:\SIFT-CNN\CNN\Queries\\" + str(i) + ".jpg"
    query = cv2.imread(querypath, cv2.IMREAD_GRAYSCALE)
    keypoints_query, descriptors_query = sift.detectAndCompute(query, None)
    descriptors_query = descriptors_query/np.linalg.norm(descriptors_query)
    if(i<10):
        querypath = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\0" + str(i) + ".npy"
    else:
        querypath = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\\" + str(i) + ".npy"
    np.save(querypath, descriptors_query)

for i in range(1, 5001):
    if(i<10):
        imagepath = r"D:\SIFT-CNN\CNN\Images\000" + str(i) + ".jpg"
    elif(i<100):
        imagepath = r"D:\SIFT-CNN\CNN\Images\00" + str(i) + ".jpg"
    elif(i<1000):
        imagepath = r"D:\SIFT-CNN\CNN\Images\0" + str(i) + ".jpg"
    else:
        imagepath = r"D:\SIFT-CNN\CNN\Images\\" + str(i) + ".jpg"
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    keypoints_img, descriptors_img = sift.detectAndCompute(img, None)
    descriptors_img = descriptors_img/np.linalg.norm(descriptors_img)
    if(i<10):
        imgpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\000" + str(i) + ".npy"
    elif(i<100):
        imgpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\00" + str(i) + ".npy"
    elif(i<1000):
        imgpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\0" + str(i) + ".npy"
    else:
        imgpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\\" + str(i) + ".npy"
    np.save(imgpath, descriptors_img)