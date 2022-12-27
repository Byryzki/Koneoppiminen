from operator import indexOf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from statistics import NormalDist

gt = []
pred = []

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

testbdict = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\test_batch")
databdict1 = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\data_batch_1")
databdict2 = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\data_batch_2")
databdict3 = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\data_batch_3")
databdict4 = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\data_batch_4")
databdict5 = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\data_batch_5")
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')

Xt = testbdict["data"]
Yt = testbdict["labels"]

X1 = databdict1["data"]
Y1 = databdict1["labels"]

X2 = databdict2["data"]
Y2 = databdict2["labels"]

X3 = databdict3["data"]
Y3 = databdict3["labels"]

X4 = databdict4["data"]
Y4 = databdict4["labels"]

X5 = databdict5["data"]
Y5 = databdict5["labels"]

Xa = np.concatenate((X1, X2, X3, X4, X5))
Ya = np.concatenate((Y1, Y2, Y3, Y4, Y5))

print(Xa.shape)

labeldict = unpickle("C:\\Users\\laine\\Dropbox\\Opinnot\\Johdanto ML\\Week 3\\cifar-10-batches-py\\batches.meta")
label_names = labeldict["label_names"]

X = Xt.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Yt)

def cifar10_color(img): #Returns means and vars of the color channels of one image
    
    xr = []
    xg = []
    xb = []

    for pxl in img:
        xr.append(img[::3])
        xg.append(img[1::3])
        xb.append(img[2::3])

    mr = np.mean(xr)
    vr = np.var(xr)
    mg = np.mean(xg)
    vg = np.var(xg)
    mb = np.mean(xb)
    vb = np.var(xb)

    return mr, mg, mb, vr, vg, vb

def cifar10_naivebayes(classvals, timg): #Calculates probabilities for each class and pick the most likely class.
    probs = []
    denom = 0

    for vals in classvals:
        cls = (NormalDist(vals[0],vals[3]).pdf(timg[0])*NormalDist(vals[1],vals[4]).pdf(timg[1])*NormalDist(vals[2],vals[5]).pdf(timg[2]))*0.1
        denom += cls

    for vals in classvals:
        prob = (NormalDist(vals[0],vals[3]).pdf(timg[0])*NormalDist(vals[1],vals[4]).pdf(timg[1])*NormalDist(vals[2],vals[5]).pdf(timg[2])*0.1)/denom
        probs.append(prob)

    print(label_names[probs.index(max(probs))-1])

    return label_names[probs.index(max(probs))-1]
    #For reasons unknown leans towards cats or automobiles in almost all training sets ://

def samplepicker(cn, X, Y): #picks a sample of only one class from training data
    sample = []

    for i, img in enumerate(X):
        if len(sample) <= 200: 
            if cn == Y[i]:
                sample.append(img)

        else:
            break

    return sample

def class_acc(pred, gt): #compares how many predictions where right

    score = 0
    for i, label in enumerate(gt):
        if pred[i] == label:
            score += 1

    acc = 100 * (score / len(gt))
    accf = "{:.2f}".format(acc)

    return f"Classification accuracy: {accf} % for {i+1} images"

def cifar10_classifier_random(x):

    rnd = random.randint(0,9)
    
    return label_names[Ya[rnd]]


#Naive_Bayes_Learn???
    #collecting class parameters

classvals = []   
for cls in range(10):
    mur = []
    mug = []
    mub = []
    varr = []
    varg = []
    varb = []

    for i, img in enumerate(samplepicker(cls,Xa,Ya)): #Käydään läpi haluttu luokka tietystä nipusta
        
        mr,mg,mb,vr,vg,vb = cifar10_color(img)

        mur.append(mr)
        mug.append(mg)
        mub.append(mb)
        varr.append(vr)
        varg.append(vg)
        varb.append(vb)

    classvals.append([np.mean(mur),np.mean(mug),np.mean(mub),np.mean(varr),np.mean(varg),np.mean(varb)])

#preparing test images

    for i in range(X.shape[0]):
        # Show some images randomly
        if random.random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")

            mr,mg,mb,vr,vg,vb = cifar10_color(X[i])
            imgvals = [mr,mg,mb,vr,vg,vb]
            
            gt.append(label_names[Y[i]])
            pred.append(cifar10_naivebayes(classvals, imgvals))

            plt.pause(0.2)

print(class_acc(pred,gt))

