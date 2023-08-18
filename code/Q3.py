import h5py as hf                           #for reading data from .mat file
import numpy as np
import matplotlib.pyplot as plt

def mean(x):                        # mean of given sample data
    return np.sum(x)/np.size(x)

def Cov2(x,y):                      # Covariance of given sample data
    x1 = x - mean(x)
    y1 = y - mean(y)
    C = np.empty((2,2))
    C[0,0] = np.sum(np.multiply(x1,x1))/np.size(x1)
    C[1,1] = np.sum(np.multiply(y1,y1))/np.size(x1)
    C[1,0] = C[0,1] = np.sum(np.multiply(x1,y1))/np.size(x1)
    return C

def m_var(x,xm,ym,s):               # returns y co_orinates corresponding to x co_ordinates,(xm,ym) point on line,s vector along the line
    m = s[1]/s[0]
    c = ym - m*xm
    y = m*x
    y = y + c
    return y

xin = input("Enter which set to show for(\n'1' for set1\n'2' for set2)\n:")

filename = ""

if xin=='1':
    filename = "points2D_Set1.mat"
elif xin=='2':
    filename = "points2D_Set2.mat"
else :
    print("invalid option")
    quit()


# loading matlab variables into python
test = hf.File('../data/'+filename,'r')         

x = test.get('x')
y = test.get('y')

x = np.array(x)
y = np.array(y)


bound = np.array([np.amin(x),np.amax(x)])

Cv = Cov2(x,y)
x_m = mean(x)
y_m = mean(y)
val,vec = np.linalg.eig(Cv)

imax = np.argmax(val)

v = vec[:,imax]             #eigenvector corresponding to larger eigenvalue

xp = np.linspace(bound[0],bound[1],100)
yp = m_var(xp,x_m,y_m,v)

plt.scatter(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Scatter plot for dataset {xin}")
plt.plot(xp,yp,'r',label="Approximated linear relationship")
plt.legend()
plt.show()
