import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 
from numpy import linalg as LA
# Reference: R. van der Merwe and E. Wan. 
# The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
#
# By Zhe Hu at City University of Hong Kong, 05/01/2017

def cholupdate(R,x,sign):
    p = len(x)
#        
    x = np.reshape(x, (p,))    
     
    for k in range(p):
                
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
#       
            
              
        c = r/R[k,k]
        s = x[k]/R[k,k]
       
        print('c =',c)
        print('s =',s)
                       
        R[k,k] = r
        
        if sign == '+':
            R[k,(k+1):p] = (R[k,(k+1):p] + s*x[(k+1):p])/c
            
        elif sign == '-':
            R[k,(k+1):p] = (R[k,(k+1):p] - s*x[(k+1):p])/c        
        
        x[(k+1):p]= c*x[(k+1):p] - s*R[k, (k+1):p]
 
    return -R

#  
    
C1 = np.array([[8,2,3],[1,3,4],[3,4,3]])
    
print('C1 = ',C1)
    
q,S = LA.qr(C1.T, 'reduced')
    
print('S = ',S)

x = np.array([1, 1, 1/np.sqrt(2)])
#x = np.reshape(x, (len(x),1))

print('x =',x)

Sm = cholupdate(S, x, '-')
Sp = cholupdate(S, x, '+')

print('S- =',Sm)

print('S+ =',Sp)
