import numpy as np
import cv2

sift_feature = []
sift_image = []

class Result:
    def __init__(self, id, dis):
        self.id = id
        self.dis = dis

def sortResults(r):
    return r.dis  

output = open(r"SIFT\ranklist.txt", "a")
sift = cv2.xfeatures2d.SIFT_create()

for i in range(1, 21):
    if(i<10):
        querysift = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\0" + str(i) + ".npy"
    else:
        querysift = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\\" + str(i) + ".npy"
    sift_feature.append(np.load(querysift))

for i in range(1, 5001):
    if(i<10):
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\000" + str(i) + ".npy"
    elif (i<100):
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\00" + str(i) + ".npy"
    elif (i<1000):
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\0" + str(i) + ".npy"
    else:
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\\" + str(i) + ".npy"
    sift_image.append(np.load(siftpath))

sift_feature = np.array(sift_feature)
sift_image = np.array(sift_image)

i=1
for q in sift_feature:
    output.write("Q"+str(i)+":")
    j=1
    results = []

    for f in sift_image:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(q, f, k=2)
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append([m])
        similarity = len(goodMatches)/(len(matches)+1)
        result = Result(j, 1-similarity)
        results.append(result)
        j+=1
    results = sorted(results, key = sortResults)
    for r in results:
        output.write(" "+str(r.id))
    output.write("\n")
    i+=1
output.close()
