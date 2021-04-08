import numpy as np
import cv2

features = []
queries = []
sift_feature = []
sift_image = []

class Result:
    def __init__(self, id, dis):
        self.id = id
        self.dis = dis

def sortResults(r):
    return r.dis  

output = open("ranklist.txt", "a")
sift = cv2.xfeatures2d.SIFT_create()

for i in range(1, 21):
    if(i<10):
        querypath = r"D:\SIFT-CNN\CNN\VGG_features\Queries\0" + str(i) + ".npy"
        querysift = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\0" + str(i) + ".npy"
    else:
        querypath = r"D:\SIFT-CNN\CNN\VGG_features\Queries\\" + str(i) + ".npy"
        querysift = r"D:\SIFT-CNN\SIFT\SIFT_features\Queries\\" + str(i) + ".npy"
    queries.append(np.load(querypath))
    sift_feature.append(np.load(querysift))

for i in range(1, 5001):
    if(i<10):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\000" + str(i) + ".npy"
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\000" + str(i) + ".npy"
    elif (i<100):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\00" + str(i) + ".npy"
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\00" + str(i) + ".npy"
    elif (i<1000):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\0" + str(i) + ".npy"
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\0" + str(i) + ".npy"
    else:
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\\" + str(i) + ".npy"
        siftpath = r"D:\SIFT-CNN\SIFT\SIFT_features\Images\\" + str(i) + ".npy"
    features.append(np.load(path))
    sift_image.append(np.load(siftpath))

queries = np.array(queries)
features = np.array(features)
sift_feature = np.array(sift_feature)
sift_image = np.array(sift_image)

i=1
for q in queries:
    output.write("Q"+str(i)+":")
    j=1
    results = []

    for f in features:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(sift_feature[i - 1], sift_image[j - 1], k=2)
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.72*n.distance:
                goodMatches.append([m])
        similarity = len(goodMatches)/len(matches)
        dis = np.linalg.norm(q-f)
        result = Result(j, 10*dis*(1-similarity))
        results.append(result)
        j+=1
    results = sorted(results, key = sortResults)
    for r in results:
        output.write(" "+str(r.id))
    output.write("\n")
    i+=1
output.close()
