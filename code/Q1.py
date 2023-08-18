import numpy as np
import matplotlib.pyplot as plt
def uniform_ellipse(n):             # generates n random points distributed uniformaly inside the given ellipse 
    rt=np.random.uniform(0,1,n)
    at=np.random.uniform(0,2*np.pi,n)
    rt=np.sqrt(rt)
    x=2*rt*np.sin(at)
    y=rt*np.cos(at)
    return (x,y)

def uniform_triangle(n):            # generates n random points distributed uniformaly inside the given triangle
    t=np.random.uniform(0,1,n)
    y=np.exp(1)*(1-np.sqrt(1-t))
    x=np.random.uniform((np.pi*y)/(3*np.exp(1)),np.pi-(2*np.pi*y)/(3*np.exp(1)))
    return (x,y)

fig,ax = plt.subplots(1,2)
e = uniform_ellipse(10000000)
t = uniform_triangle(10000000)

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].hist2d(t[0],t[1],bins=50)
ax[1].hist2d(e[0],e[1],bins=50)

ax[0].set_title("Uniform distribution on triangle")
ax[1].set_title("Uniform distribution on ellipse")

plt.show()
    
    
    
                    
    
    
    
    
    
    
    
    
