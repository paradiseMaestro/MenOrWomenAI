import os
import cv2
import numpy as np

# from PIL import Image
# from PIL import ImageChops

# image_one = Image.open('1.jpg')
# image_two = Image.open('0.jpg')

# diff = ImageChops.difference(image_one, image_two)

# if diff.getbbox():
#     print("no")
# else:
#     print("yes")


# q = os.listdir('w')

# i = 0
# while( i < len(q)-1):
#     # img = cv2.imread(q[i], cv2.IMREAD_UNCHANGED)
#     image_one = Image.open(q[i])
#     t = 0
#     W = os.listdir('Women')
#     while( t < len(t)-1):
#         t = t + 1
#     cv2.imwrite('./Women/'+str(i)+'.jpg',img)














# rename files
import os as s




# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------

def renemaFilesInNumbers(from_,to_):
    i = 0
    name1 = s.listdir('./'+from_)
    while(i < len(name1)):
    # while(i < 101):
        s.rename('C:/Users/79042/Pictures/q/'+from_+'/' +name1[i],'./'+to_+'/'+ ''+str(i)+'.jpg')
        i = i + 1 

# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------

from PIL import Image
import PIL.ImageOps

def invert(g,b):

    image = Image.open(g)

    inverted_image = PIL.ImageOps.invert(image)

    inverted_image.save(b)

# import cv2
import cv2
def ResizeImags(from_,to_):
    i = 0
    name1 = s.listdir('C:/Users/79042/Pictures/q/'+from_)
    while( i < len(name1)):
        # invert('C:/Users/79042/Pictures/q/'+from_+'/'+str(i)+'.jpg','./'+to_+'/'+str(i)+'.jpg')

        img = cv2.imread('C:/Users/79042/Pictures/q/'+from_+'/'+str(i)+'.jpg', cv2.IMREAD_UNCHANGED)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_image, (299, 299), interpolation = cv2.INTER_AREA)
        # res = cv2.flip(img, 1)
        cv2.imwrite('./'+to_+'/'+str(i)+'.jpg',resized)
        # cv2.imwrite('./'+to_+'/'+str(i)+'_.jpg',res)
        i = i + 1

# ResizeImags('b','v')


import os
import cv2
import numpy as np

from PIL import Image
from PIL import ImageChops

def deleteDubl(pg1,pg2):

    def IsTrue(im1,im2):
        image_one = Image.open(im1)
        image_two = Image.open(im2)

        diff = ImageChops.difference(image_one, image_two)
        IsTrue = True
        if diff.getbbox():
            IsTrue = False
        else:
            IsTrue = True
        return IsTrue
    q = os.listdir(pg1)
    i = 0
    while(True):
        w = 1
        q = os.listdir(pg1)
        if (len(q) == 0):
            break
        while( w < len(q)):
            if( IsTrue( pg1+'/'+q[0],pg1+'/'+q[w] ) ):
                os.remove(pg1+'/'+q[w])
            w = w + 1
        i = i + 1
        # if (len(q) == 1):
            # break
        os.rename('./'+pg1+'/'+q[0],'./'+pg2+'/'+str(i)+'.jpg')

