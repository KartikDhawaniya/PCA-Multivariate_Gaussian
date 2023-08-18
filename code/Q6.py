import numpy as np
from numpy.lib.function_base import cov
from scipy.sparse import linalg as sp
import PIL as pl                                # for loading each image data
import os
import matplotlib.pyplot as plt


def dim_reduce(img,basis,mean):                 # function to approximate img in smaller basis
    img_s = img - mean
    c = np.matmul(img_s,basis)
    return c


path = "../data/data_fruit"
imgs = np.empty((16,19200))                     # stores each image data          

i = 0

ext = ".png"

for file in os.listdir(path):
    if os.path.splitext(file)[1] == ext:
        imgs[i]=np.reshape(np.array(pl.Image.open(os.path.join(path,file))),(19200))
        i+=1

# print(imgs[0])

mean = np.sum(imgs,axis=0)/16                           # mean of image data
n_img = imgs-mean
C = np.matmul(np.transpose(imgs-mean),imgs-mean)/16     # Covariance of image data

# C = np.array([[1,0],[0,1]])

val,vec = sp.eigsh(C,k=10)                              # top 10 eigenvalues and its corresponding eigen vectors

x = input("Enter command\n 'g' for Graph of top 10 eigenvalues\n 'i' for image representation of mean and eigenvectors\n 'c' for closest representation of each fruit\n 'n' for new fruits generated\n :")

if(x=="i"):
    figure, axis = plt.subplots(1,5)
    plt.suptitle("Image representation of mean and eigenvectors")
    axis[0].imshow(np.reshape(mean/255,(80,80,3)))
    axis[0].set_title("Mean")
    for i in range(1,5):
        v = np.reshape(np.real(vec[:,::-1][:,i-1])-np.amin(np.real(vec[:,::-1][:,i-1])),(80,80,3))      # scaling and shifting of eigenvectors
        v = v/np.amax(v)                                                                                # scaling and shifting of eigenvectors
        axis[i].imshow(v)
        axis[i].set_title(f"v{i}")
    plt.show()

elif x=="g":
    plt.plot(val[::-1])
    plt.xlabel("indices of eigenvalues")
    plt.ylabel("Eigenvalues")
    plt.title("Graph for top 10 eigenvalues")
    plt.show()

elif x=="c" :
    figure, axis = plt.subplots(4,8)
    plt.suptitle("Approximation of fruit images")
    v = vec[:,::-1][:,:4]
    for i in range(16):
        c = dim_reduce(np.transpose(imgs[i]),v,np.transpose(mean))
        n_img = mean + np.matmul(v,np.transpose(c))
        n_img = n_img - np.amin(n_img)
        n_img = n_img/np.amax(n_img)
        axis[2*(i//8)][i%8].imshow(np.reshape(imgs[i]/255,(80,80,3)))
        axis[2*(i//8)][i%8].set_title("Original")
        axis[1+2*(i//8)][i%8].imshow(np.reshape(n_img,(80,80,3)))
        axis[1+2*(i//8)][i%8].set_title("Approximated")
    plt.show()

elif x=='n' :
    cd = np.empty((16,4))
    figure, axis = plt.subplots(1,3)
    plt.suptitle("3 new fruits")
    v = vec[:,::-1][:,:4]
    l = val[::-1][:4]
    c1=c2=c3=c4=0.5
    i = mean + c1*np.sqrt(l[0])*v[:,0] + c2*np.sqrt(l[1])*v[:,1] + c3*np.sqrt(l[2])*v[:,2] + c4*np.sqrt(l[3])*v[:,3]
    j = mean - c1*np.sqrt(l[0])*v[:,0] - c2*np.sqrt(l[1])*v[:,1] - c3*np.sqrt(l[2])*v[:,2] - c4*np.sqrt(l[3])*v[:,3]
    axis[0].imshow(np.reshape(mean/255,(80,80,3)))
    axis[0].set_title("Fruit1")
    axis[1].imshow(np.reshape(i/255,(80,80,3)))
    axis[1].set_title("Fruit2")
    axis[2].imshow(np.reshape(j/255,(80,80,3)))
    axis[2].set_title("Fruit3")

    plt.show()

else:
    print("Incorrect command")

