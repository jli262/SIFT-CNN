import numpy as np

features = []
queries = []

class Result:
    def __init__(self, id, dis):
        self.id = id
        self.dis = dis

def sortResults(r):
    return r.dis  

output = open("ranklist.txt", "a")

for i in range(1, 21):
    if(i<10):
        querypath = r"VGG_features\Queries\0" + str(i) + ".npy"
    else:
        querypath = r"VGG_features\Queries\\" + str(i) + ".npy"
    queries.append(np.load(querypath))

for i in range(1, 5001):
    if(i<10):
        path = r"VGG_features\Images\000" + str(i) + ".npy"
    elif (i<100):
        path = r"VGG_features\Images\00" + str(i) + ".npy"
    elif (i<1000):
        path = r"VGG_features\Images\0" + str(i) + ".npy"
    else:
        path = r"VGG_features\Images\\" + str(i) + ".npy"
    features.append(np.load(path))

queries = np.array(queries)
features = np.array(features)

i=1
for q in queries:
    output.write("Q"+str(i)+":")
    i+=1
    j=1
    results = []
    for f in features:
        dis = np.linalg.norm(q-f)
        result = Result(j, dis)
        results.append(result)
        j+=1
    results = sorted(results, key = sortResults)
    for r in results:
        output.write(" "+str(r.id))
    output.write("\n")
output.close()
