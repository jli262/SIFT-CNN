from PIL import Image
import os
from numpy import *
from pylab import *

i=1
params="--edge-thresh 10 --peak-thresh 5"
for i in range(1, 5001):
    if(i<10):
        imagename = r"C:\Users\Z0047kke\cs4186\Images\000" + str(i) + ".jpg"
    elif (i<100):
        imagename = r"C:\Users\Z0047kke\cs4186\Images\00" + str(i) + ".jpg"
    elif (i<1000):
        imagename = r"C:\Users\Z0047kke\cs4186\Images\0" + str(i) + ".jpg"
    else:
        imagename = r"C:\Users\Z0047kke\cs4186\Images\\" + str(i) + ".jpg"
    im = Image.open(imagename)
    if(i<=2000):
        if(i<10):
            txtpath = r"C:\Users\Z0047kke\cs4186\Images\000" + str(i) + ".txt"
        elif (i<100):
            txtpath = r"C:\Users\Z0047kke\cs4186\Images\00" + str(i) + ".txt"
        elif (i<1000):
            txtpath = r"C:\Users\Z0047kke\cs4186\Images\0" + str(i) + ".txt"
        else:
            txtpath = r"C:\Users\Z0047kke\cs4186\Images\\" + str(i) + ".txt"
        f = open(txtpath, "r")
        strr = f.readline().split(" ")
        f.close()
        im = im.crop((int(strr[0]), int(strr[1]), int(strr[0])+int(strr[2]), int(strr[1])+int(strr[3])))
    im = array(im.convert('L'))
    if(i<10):
        path = r"C:\Users\Z0047kke\cs4186\Images\000" + str(i) + ".sift"
    elif (i<100):
        path = r"C:\Users\Z0047kke\cs4186\Images\00" + str(i) + ".sift"
    elif (i<1000):
        path = r"C:\Users\Z0047kke\cs4186\Images\0" + str(i) + ".sift"
    else:
        path = r"C:\Users\Z0047kke\cs4186\Images\\" + str(i) + ".sift"
    if (imagename[-3:] != 'pgm'):
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
    imagename = 'tmp.pgm'
    cmd = "D:\\UserData\\Z0047kke\\vlfeat-0.9.20\\vlfeat-0.9.20\\bin\\win64\\sift.exe "+imagename+" --output="+path+" "+params
    os.system(cmd)