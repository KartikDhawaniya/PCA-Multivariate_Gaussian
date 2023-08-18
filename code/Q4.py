import numpy as np
import h5py as hf                           # for reading data from .mat file
import matplotlib.pyplot as plt
import scipy.io as sio


# loading matlab variables into python
filename = '../data/mnist.mat'
file = hf.File(filename,'r')
labels_train = file.get('labels_train')
digits_train = file.get('digits_train')

labels_train = np.array(labels_train)
digits_train = np.array(digits_train)

num = []                # list of numpy arrays each storing vectors for images of digits
mean = []               # list of mean vectors corresponding to each digit image
Cov = []                # list of covariance matrices corresponding to each digit image
eig_val = []            # list of eigenvalues corresponding to each digit image
eig_vec = []            # list of eigenvectors corresponding to each digit image
p_val = []              # lambda1 for each digit
p_vec = []              # v1 for each digit
pmv1 = []
pmv2 = []

# separating data for each digit
for i in range(10):
    t = np.where(labels_train[0]==i)
    temp = digits_train[t[0],:,:]
    temp = np.reshape(temp,(temp.shape[0],28*28,1))
    num.append(temp)

# for computing mean,Cov,eigenvalues,eigenvectors,lambda1 and v1
for i in range(10):
    mn = np.sum(num[i],axis=0)/num[i].shape[0]
    mean.append(mn)
    temp = num[i] - mn
    temp = np.reshape(temp,(num[i].shape[0],784))
    C = np.matmul(np.transpose(temp),temp)/num[i].shape[0]
    Cov.append(C)
    val,vec = np.linalg.eig(C)
    imax = np.argmax(np.real(val))
    p_val.append(np.real(val[imax]))
    v = np.real(vec[:,imax])
    p_vec.append(v)
    p = mn + np.sqrt(np.real(val[imax]))*v
    q = mn - np.sqrt(np.real(val[imax]))*v
    pmv1.append(p)
    pmv2.append(q)
    eig_val.append(val)
    eig_vec.append(vec)


# saving computed variables to Q4_data.mat in results folder
sio.savemat("../results/Q4_data.mat",{"mean":mean, "covariance":Cov, "lambda1": p_val, "v1":p_vec, "principal mode of variation+": pmv1, "principal mode of variation-": pmv2})
p_val = p_vec = 0
xin = input("Enter command\n 'm' for mean images of each digit\n 'g' for plot of eigenvalues\n '3' for 3 images mentioned for each digit in question\n :")

if xin=="m":
    fig,ax = plt.subplots(2,5)
    plt.suptitle("Images of mean for each digit") 
    for i in range(10):
        ax[i//5][i%5].imshow(np.transpose(np.reshape(mean[i],(28,28))))
        ax[i//5][i%5].set_title(i,fontsize=24,color='blue')
    plt.show()

elif xin=="g":
    x = np.arange(1,28*28+1)
    fig,ax = plt.subplots(2,5,constrained_layout=True)
    plt.suptitle("Graphs of sorted eigenvalues")
    for i in range(10):
        ax[i//5][i%5].plot(x,np.sort(np.real(eig_val[i]))[::-1])
        ax[i//5][i%5].set_title(i,fontsize=24,color='blue')
        ax[i//5][i%5].set_xlabel("Sorted order of Eigenvalues",color='green')
        ax[i//5][i%5].set_ylabel("Value",color='green')
    # plt.tight_layout()
    plt.show()
        # print(np.sum(eig_val[i]))

elif xin=="3":
    figure, axis = plt.subplots(5, 6, constrained_layout=True)
    for i in range(10):
        m = mean[i]
        l1 = np.sqrt(np.amax(np.real(eig_val[i])))
        imax = np.argmax(np.real(eig_val[i]))
        v1 = np.reshape(np.real(eig_vec[i][:,imax]),(784,1))
        m1 = m - l1*v1                                                      # μ-√(λ1)*v1
        m2 = m + l1*v1                                                      # μ+√(λ1)*v1
        axis[i//2][(i%2)*3].imshow(np.transpose(np.reshape(m1,(28,28))))
        axis[i//2][(i%2)*3].set_title("μ-√(λ1)*v1",color='blue')   
        axis[i//2][(i%2)*3+1].imshow(np.transpose(np.reshape(m,(28,28))))
        axis[i//2][(i%2)*3+1].set_title("μ",color='blue') 
        axis[i//2][(i%2)*3+2].imshow(np.transpose(np.reshape(m2,(28,28))))
        axis[i//2][(i%2)*3+2].set_title("μ+√(λ1)*v1",color='blue') 
    plt.show()

else :
    print("Incorrect option")