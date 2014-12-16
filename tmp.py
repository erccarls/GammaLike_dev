
import numpy as np
gamma=1.26
r_s=20.0
#offset=(0, 0)
#rotation=0.0
#axesratio=1

#sinAng = np.sin(np.deg2rad(rotation))
#cosAng = np.cos(np.deg2rad(rotation))
    
def func(x,y,z):
    #x = X
    #y = ((Y-offset[0])*cosAng+(Z-offset[1])*sinAng) / axesratio
    #z = -(Y-offset[0])*sinAng+(Z-offset[1])*cosAng

    r=np.sqrt(x*x+y*y+z*z)
    return r**-gamma*(1/(1+r/r_s)**(3-gamma))
    