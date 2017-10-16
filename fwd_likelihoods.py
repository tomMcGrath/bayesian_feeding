import numpy as np
from scipy import integrate, optimize
import ODEs

"""
Rate functions & transition kernel
"""
def phi_const(theta):
    ans = theta
    return ans

def phi_F(t, x0, rate, theta1):
    """
    Linear rate function based on stomach fullness
    """
    x = rate*t + x0

    ans = theta1

    return ans
"""
def phi_L_quad(t, x0, k1, theta7, theta8):
    if x0 > 0.0:
        t_c = 2.0*np.sqrt(x0)/k1 # time to hit zero
    else:
        t_c = 0.0

    if t >= t_c:
        x = 0.0
    else:
        x = 0.25*np.power((2.*np.sqrt(x0) - k1*t), 2)

    ans = np.power(1./(theta7 + theta8*x), 2)

    return ans
"""

def phi_L(t, x0, k1, theta7, theta8):
    if x0 > 0.0:
        t_c = 2.0*np.sqrt(x0)/k1 # time to hit zero
    else:
        t_c = 0.0

    if t >= t_c:
        x = 0.0
    else:
        x = 0.25*np.power((2.*np.sqrt(x0) - k1*t), 2)

    ans = 1./(theta7 + theta8*x)

    return ans

def Q(x, theta5, theta6):
    eps = 0.01
    ans = eps + (1. - 2.*eps)/(1. + np.exp(-0.1*theta5*(x-20.*theta6)))

    return ans

"""
Rate function integrals
"""
def psi_const(t, theta):
    ans = theta*t
    return ans

def psi_F(t, x0, rate, theta1):
    ans = integrate.quad(phi_F, 0, t, args=(x0, rate, theta1))[0]
    return ans

def psi_L(t, x0, k1, theta7, theta8):
    ans = integrate.quad(phi_L, 0, t, args=(x0, k1, theta7, theta8))[0]
    return ans

def L_an_int(t, x0, k1, theta7, theta8):
    return 2.*np.arctan(0.5*np.sqrt(theta8/theta7)*(k1*t - 2.*np.sqrt(x0)))/(k1*np.sqrt(theta7*theta8))

def psi_L_an(t, x0, k1, theta7, theta8):
    t_c = 2.*np.sqrt(x0)/k1

    if t <= t_c:
        return L_an_int(t, x0, k1, theta7, theta8) - L_an_int(0, x0, k1, theta7, theta8)

    else:

        return psi_L_an(t_c, x0, k1, theta7, theta8) + (t - t_c)/theta7    

"""
def L_an_int(t, x0, k1, theta7, theta8):
    x = 0.25*np.power(k1*t, 2) - k1*t*np.sqrt(x0) + x0

    part1 = k1*t*theta8*np.power(theta7, 2)*(0.5*theta7 + theta8*(0.25*k1*t*np.sqrt(x0) - 0.5*x0))
    part2_1 = theta7*(np.power(theta7, 2) + theta7*theta8*(x + x0) + x0*np.power(theta8, 2)*x)
    part2_2 = np.sqrt(theta7*theta8)*(np.arctan((0.5*k1*t - np.sqrt(x0))*np.sqrt(theta8/theta7)) + np.arctan(np.sqrt(theta8*x0/theta7)))
    
    denom = k1*np.power(theta7, 3)*theta8*(theta7 + x0*theta8)*(theta7 + theta8*x)

    ans = (part1 + part2_1*part2_2)/denom
    return ans

def psi_L_an(t, x0, k1, theta7, theta8):
    t_c = 2.*np.sqrt(x0)/k1

    if t <= t_c:
        return L_an_int(t, x0, k1, theta7, theta8)

    else:
        return psi_L_an(t_c, x0, k1, theta7, theta8) + (t - t_c)/np.power(theta7, 2)
"""
"""
CDFs and inversions for sampling
"""
def CDF_F(t, x0, rate, theta1):
    psi = psi_F(t, x0, rate, theta1)
    ans = 1. - np.exp(-psi)
    return ans

def CDF_L(t, x0, k1, theta8, theta9):
    psi = psi_L(t, x0, k1, theta8, theta9)
    ans = 1. - np.exp(-psi)
    return ans

def PDF_F(t, x0, rate, theta1):
    psi = psi_F(t, x0, rate, theta1)
    phi = phi_F(t, x0, rate, theta1)
    ans = phi*np.exp(-psi)
    return ans

def PDF_L(t, x0, k1, theta8, theta9):
    psi = psi_L_an(t, x0, k1, theta8, theta9)
    phi = phi(t, x0, k1, theta8, theta9)
    ans = phi*np.exp(-psi)
    return ans

def CDF_inv_F(u, x0, rate, theta1):
    def f_to_min(t, u, x0, rate, theta1):
        ans = CDF_F(t, x0, rate, theta1) - u
        return ans
    
    ans = optimize.brentq(f_to_min, 0.0, 1e5, args=(u, x0, rate, theta1))
    return ans
"""
def CDF_inv_L(u, x0, k1, theta8, theta9):
    def f_to_min(t, u, x0, k1, theta8, theta9):
        ans = CDF_L(t, x0, k1, theta8, theta9) - u
        return ans
    #print f_to_min(0, u, x0, k1, theta8, theta9), f_to_min(1e12, u, x0, k1, theta8, theta9)
    ans = optimize.brentq(f_to_min, 0.0, 1e12, args=(u, x0, k1, theta8, theta9))
    return ans
"""

def CDF_inv_L(u, x0, k1, theta7, theta8):
    def f_to_min(t, u, x0, k1, theta7, theta8):
        ans = np.log(1-u) + psi_L_an(t, x0, k1, theta7, theta8)
        return ans

    try:
        ans = optimize.brentq(f_to_min, 0.0, 1e12, args=(u, x0, k1, theta7, theta8))
    except RuntimeError:
        ans = optimize.brentq(f_to_min, 0.0, 1e14, maxiter=1000, args=(u, x0, k1, theta7, theta8))

    return ans

def sample_F(x0, rate, theta1):
    u = np.random.uniform(0,1)
    ans = CDF_inv_F(u, x0, rate, theta1)
    return ans

def sample_L(x0, k1, theta7, theta8):
    u = np.random.uniform(0,1)
    ans = CDF_inv_L(u, x0, k1, theta7, theta8)
    return ans

"""
ODEs
"""
def feeding_ode(x, t, rate):
    """
    ODE for feeding with rate rate
    """  
    calorie_content = 3.5
    return calorie_content*rate
    
def digestion_ode(x, t, k1):
    """
    ODE for digestion with rate k1
    """
    if x > 0:
        ans = -k1*np.sqrt(x)
    else:
        ans = 0.0
    return ans