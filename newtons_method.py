'''
This module calculates Newton's Method for solving non-linear systems of equations.
See https://en.wikipedia.org/wiki/Newton%27s_method for more info.

Functions: 
    - newtons_method(F, J, x_guess, tol=1e-5)
      Solves non-linear systems of the form F(x) = 0
      where J is the Jacobian of F.

Parameters:
    - F (callable): a function representing the non-linear system of equations
    - J (callable): a function represent the Jacobian of F
    - x_guess (array-like): initial guess for the vector x that solves F
    - tol (float, optional): the desired tolerance level, default value 1e-5

Returns:
    - x: the approximated root x* that solves F to our desired tolerance
    - iterations: the number of iterations to solve the non-linear system

Author: Aravind Adalarasu
'''


import numpy as np

def newtons_method(F, J, x_guess, tol=1e-5):
    '''Function that implements Newton's Method for Non-linear system. This function
    returns a numpy array x, which is the approximated solution to the problem F(x)=0.'''

    x = np.array(x_guess)
    iterations = 0

    while True:
        F_val = F(x)
        J_val = J(x)        
        delta = np.linalg.solve(J_val, F_val)   # finding J inverse is costly, 
        x = x - delta                           # so I use this method to solve for delta instead
        iterations += 1
        if np.linalg.norm(F_val) < tol:
            return x, iterations
    
