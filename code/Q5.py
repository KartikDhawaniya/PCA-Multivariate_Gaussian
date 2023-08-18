import numpy as np
import h5py as hf
import matplotlib.pyplot as plt

def dim_reduce(img,basis,mean):                         # funtion to compute 84 coordinates in 84-dimensional basis
    img_s = img - mean
    c = np.matmul(img_s,basis)
    return c

np.random.seed(0)

filename = '../data/mnist.mat'
file = hf.File(filename,'r')
labels_train = file.get('labels_train')
digits_train = file.get('digits_train')

labels_train = np.array(labels_train)
digits_train = np.array(digits_train)

num = []                            # list of numpy arrays each storing vectors for images of digits
mean = []                           # list of mean vectors corresponding to each digit image
Cov = []                            # list of covariance matrices corresponding to each digit image
red_dim = []                        # 84-dimensional  basis for each digit

# separating data for each digit
for i in range(10):
    t = np.where(labels_train[0]==i)
    temp = digits_train[t[0],:,:]
    temp = np.reshape(temp,(temp.shape[0],28*28,1))
    num.append(temp)

# for computing mean,Cov and basis
for i in range(10):
    mn = np.sum(num[i],axis=0)/num[i].shape[0]
    mean.append(mn)
    temp = num[i] - mn
    temp = np.reshape(temp,(num[i].shape[0],784))
    C = np.matmul(np.transpose(temp),temp)/num[i].shape[0]
    Cov.append(C)
    val,vec = np.linalg.eig(C)
    dim = np.real(vec)[:,np.real(val).argsort()[::-1][:84]]
    red_dim.append(dim)

figure, axis = plt.subplots(4,5,  constrained_layout=True)
plt.suptitle("Reconstructed images for each digit",color="blue")
for i in range(10):
    image = num[i][10]
    c = dim_reduce(np.transpose(image),red_dim[i],np.transpose(mean[i]))                # calculating 84 coordinates

    n_image = mean[i] + np.matmul(red_dim[i],np.transpose(c))                           # reconstructed image

    
    axis[2*(i//5)][i%5].imshow(np.transpose(np.reshape(image,(28,28))))
    axis[2*(i//5)][i%5].set_title("Original",color="green")
    axis[1+2*(i//5)][i%5].imshow(np.transpose(np.reshape(n_image,(28,28))))
    axis[1+2*(i//5)][i%5].set_title("Reduced",color='green')
plt.show()





