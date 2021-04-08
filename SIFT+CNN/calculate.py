import numpy as np
import cv2

features = []
queries = []

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
    else:
        querypath = r"D:\SIFT-CNN\CNN\VGG_features\Queries\\" + str(i) + ".npy"
    queries.append(np.load(querypath))

for i in range(1, 5001):
    if(i<10):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\000" + str(i) + ".npy"
    elif (i<100):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\00" + str(i) + ".npy"
    elif (i<1000):
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\0" + str(i) + ".npy"
    else:
        path = r"D:\SIFT-CNN\CNN\VGG_features\Images\\" + str(i) + ".npy"
    features.append(np.load(path))

queries = np.array(queries)
features = np.array(features)

i=1
for q in queries:
    output.write("Q"+str(i)+":")
    i+=1
    j=1
    results = []
    if(i<10):
        querypath = r"D:\SIFT-CNN\CNN\Queries\0" + str(i) + ".jpg"
    else:
        querypath = r"D:\SIFT-CNN\CNN\Queries\\" + str(i) + ".jpg"
    query = cv2.imread(querypath)
    query = cv2.cvtcolor(query, cv2.COLOR_BGR2GRAY)
    keypoints_query, descriptors_query = sift.detectAndCompute(query, None)

    for f in features:
        if(i<10):
            path = r"D:\SIFT-CNN\CNN\Images\000" + str(i) + ".jpg"
        elif (i<100):
            path = r"D:\SIFT-CNN\CNN\Images\00" + str(i) + ".jpg"
        elif (i<1000):
            path = r"D:\SIFT-CNN\CNN\Images\0" + str(i) + ".jpg"
        else:
            path = r"D:\SIFT-CNN\CNN\Images\\" + str(i) + ".jpg"

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_query,descriptors, k=2)
        goodMatches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append([m])

        dis = np.linalg.norm(q-f)
        result = Result(j, 100*dis/len(goodMatches))
        results.append(result)
        j+=1
    results = sorted(results, key = sortResults)
    for r in results:
        output.write(" "+str(r.id))
    output.write("\n")
output.close()
