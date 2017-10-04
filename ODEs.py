import numpy as np

"""
Now integrate the ODEs
"""

def feeding_ode(x, t, k2):
    """
    ODE for feeding with rate k2
    """
    calorie_content = 3.5 # kcal/gram        

    return k2*calorie_content

def digestion_ode(x, t, k1):
    """
    ODE for feeding with rate k1
    """
    if x > 0:
        ans = -k1*np.sqrt(x)
    else:
        ans = 0.0
    return ans