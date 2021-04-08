from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

VGGModel = VGG16(weights='imagenet')
model = Model(inputs=VGGModel.input, outputs=VGGModel.get_layer('fc1').output)

# for i in range(1, 5001):
#     if(i<10):
#         imagename = r"Images\000" + str(i) + ".jpg"
#     elif (i<100):
#         imagename = r"Images\00" + str(i) + ".jpg"
#     elif (i<1000):
#         imagename = r"Images\0" + str(i) + ".jpg"
#     else:
#         imagename = r"Images\\" + str(i) + ".jpg"
#     img = Image.open(imagename)
#     img = img.resize((224, 224))
#     img = img.convert('RGB')
#     imgdata = image.img_to_array(img)
#     imgdata = np.expand_dims(imgdata, axis = 0)
#     imgdata = preprocess_input(imgdata)
#     result = model.predict(imgdata)[0]
#     result = result/np.linalg.norm(result)
#     if(i<10):
#         path = r"VGG_features\Images\000" + str(i) + ".npy"
#     elif (i<100):
#         path = r"VGG_features\Images\00" + str(i) + ".npy"
#     elif (i<1000):
#         path = r"VGG_features\Images\0" + str(i) + ".npy"
#     else:
#         path = r"VGG_features\Images\\" + str(i) + ".npy"
#     np.save(path, result)


for i in range(1, 21):
    if(i<10):
        queryname = r"Queries\0" + str(i) + ".jpg"
    else:
        queryname = r"Queries\\" + str(i) + ".jpg"
    query = Image.open(queryname)
    query = query.resize((224, 224))
    query = query.convert('RGB')
    querydata = image.img_to_array(query)
    querydata = np.expand_dims(querydata, axis = 0)
    queryata = preprocess_input(querydata)
    queryresult = model.predict(querydata)[0]
    queryresult = queryresult/np.linalg.norm(queryresult)
    if(i<10):
        querypath = r"VGG_features\Queries\0" + str(i) + ".npy"
    else:
        querypath = r"VGG_features\Queries\\" + str(i) + ".npy"
    np.save(querypath, queryresult)
