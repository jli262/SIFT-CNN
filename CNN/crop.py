from PIL import Image
import os
from numpy import *
from pylab import *

i=1

for i in range(1, 21):
    if(i<10):
        imagename = r"Queries\0" + str(i) + ".jpg"
    else:
        imagename = r"Queries\\" + str(i) + ".jpg"
    im = Image.open(imagename)
    if(i<10):
        txtpath = r"Queries\0" + str(i) + ".txt"
    else:
        txtpath = r"Queries\\" + str(i) + ".txt"
    f = open(txtpath, "r")
    strr = f.readline().split(" ")
    f.close()
    im = im.crop((int(strr[0]), int(strr[1]), int(strr[0])+int(strr[2]), int(strr[1])+int(strr[3])))
    im.save(imagename)