import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 

from numpy import linalg as LA
from numpy import zeros_like, empty_like, sqrt, spacing
from scipy.linalg import LinAlgError


def algoB(R, z, lower=False):
    
    """
    Cholesky rank-1 downdate algorithm taken from [1]_ such that
    ``D'D = R'R - zz'``

    Parameters
    ----------
    R : (N, N) ndarray
        The 2D square array from which the triangular part will be used to
        read the Cholesky factor.The remaining parts are ignored.
    z : (N,) ndarray
        Downdate array
    lower: bool, optional
        Whether Cholesky factor is upper or lower triangular Cholesky
        factorization.  Default is upper-triangular.
    overwrite_R: bool, optional
        If set to True ....
    overwrite_z : bool, optional
        If set to True the entries of the input array will be modified during
        the computations instead of creating a new array.

    Returns
    -------
    D : (N, N) ndarray
        The resulting downdated Cholesky factor.

    References
    ----------
    .. [1] DOI:10.1007/BF01933218
    """


# Reference: R. van der Merwe and E. Wan. 
# The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
#
# By Zhe Hu at City University of Hong Kong, 05/01/2017

 # Just assuming arrayness, shape checks and overwrites etc. are handled.
    # Then... (with overwriting z)

    n = R.shape[0]
    eps = n * spacing(1.)  # For complex this needs modification
    alpha, beta = empty_like(z), empty_like(z)
    alpha[-1], beta[-1] = 1., 1.
    D = R.copy()

    for r in range(n):
        a = z[r] / R[r, r]
        alpha[r] = alpha[r-1] - a ** 2
        # Numerically zero or negative
        if alpha[r] < eps:
            # Made up err msg.
            raise LinAlgError('The Cholesky factor becomes nonpositive'
                              'with this downdate at the step {}'.format(r))
        beta[r] = sqrt(alpha[r])
        z[r+1:] -= a*R[r, r+1:]
        D[r, r:] *= beta[r] / beta[r-1]
        D[r, r+1:] -= a/(beta[r] * beta[r-1])*z[r+1:]

    return D

#  
    
C1 = np.array([[8,2,3],[1,3,4],[3,4,3]])
    
print('C1 = ',C1)
    
q,S = LA.qr(C1.T, 'reduced')
    
print('S = ',S)

x = np.array([1, 1, 1/np.sqrt(2)])
#x = np.reshape(x, (len(x),1))

print('x =',x)

Sm = algoB(S, x, lower = False)
Sp = algoB(S, x, lower = True)
#Sm = cholupdate(S, x, '-')
#Sp = cholupdate(S, x, '+')

print('S- =',Sm)

print('S+ =',Sp)
