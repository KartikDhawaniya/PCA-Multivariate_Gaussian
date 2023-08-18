import numpy as np
import matplotlib.pyplot as plt

def multi_normal(n,C,mu):               # generates n random points belonging to given bivariate gaussian
    mu = np.repeat(mu,n,axis=1)
    w,v = np.linalg.eig(C)
    S = np.zeros([2,2],dtype = float)
    S[0][0]=w[0]
    S[1][1]=w[1]
    S = np.sqrt(S)
    A = np.matmul(v,S)
    W = np.random.normal(0,1,(2,n))
    X = mu + np.matmul(A,W)

    return X

def ML_mean(X,n):                       # ML estimate of mean
    return np.sum(X,axis=1)/n

def ML_C(X,n):                          # ML estimate of Covariance
    m = ML_mean(X,n)
    C_ml = np.zeros([2,2],dtype=float)
    a = X - np.repeat(m.reshape((2,1)),n,axis=1)
    for i in range(2):
        for j in range(2):
            C_ml[i][j] = np.sum(np.multiply(a[i,:],a[j,:]))/(n-1)

    return C_ml

np.random.seed(0)

C = np.array([[1.625,-1.9486],[-1.9486,3.875]])
mean = np.array([[1],[2]])

x = input("Enter command(\n'm' for boxplot of error in mean\n'c' for boxplot of error in covariance\n's' for scatter plot)\n: ")

if x=="m":
    error_m = np.empty((5,100))                     # array storing errors in mean for each N

    for i in range(5):
        for j in range(100):
            a = ML_mean(multi_normal(10**(i+1),C,mean),10**(i+1))               # ML estimate of mean
            a = np.sqrt(np.sum(np.square(a.reshape((2,1))-mean)))
            error_m[i][j]=a/np.sqrt(np.sum(np.square(mean)))                    # error 

    plt.boxplot(np.transpose(error_m))              # boxplot
    plt.xlabel("log10(N)")
    plt.ylabel("Errors")
    plt.title("Box plot of errors in mean")
    plt.show()

elif x=="c":
    error_C = np.empty((5,100))                     # array storing errors in Covariance for each N

    for i in range(5):
        for j in range(100):
            b = ML_C(multi_normal(10**(i+1),C,mean),10**(i+1))                  # ML estimate of Covariance
            b = np.sqrt(np.sum(np.square(b-C)))
            error_C[i][j]=b/np.sqrt(np.sum(np.square(C)))                       # error

    plt.boxplot(np.transpose(error_C))                                          # boxplot
    plt.xlabel("log10(N)")
    plt.ylabel("Errors")
    plt.title("Box plot of errors in Covariance")
    plt.show()

elif x=="s":
    fig,ax = plt.subplots(1,5)
    plt.suptitle("Scatter plots of sample data for each N")
    # plt.xlim([-3,6])
    for i in range(1,6):
        X = multi_normal(10**i,C,mean)                  # 10^i random sample points from given 2d gaussian
        m = ML_mean(X,10**i)                            # corresponding ML estimate of mean
        c = ML_C(X,10**i)                               # corresponding ML estimate of Covariance
        val,vec = np.linalg.eig(c)                      # eigen  values and corresponding eigenvectors 
        ax[i-1].set_aspect('equal')
        ax[i-1].scatter(X[0],X[1],marker='.')
        ax[i-1].plot([m[0],m[0]+np.sqrt(val[0])*vec[0][0]],[m[1],m[1]+np.sqrt(val[0])*vec[1][0]],'orange', label='smaller mode of variation')
        ax[i-1].plot([m[0],m[0]+np.sqrt(val[1])*vec[0][1]],[m[1],m[1]+np.sqrt(val[1])*vec[1][1]],'r', label='higher mode of variation')
        ax[i-1].set_title(f"N={10**i}")
    plt.legend()    
    plt.show()
