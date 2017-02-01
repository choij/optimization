import numpy as np
import numpy.linalg as LA
import sympy
from sympy import symbols, hessian, Function, N, diff

import matplotlib.pyplot as plt

def rosenbrock():
  x, y = symbols('x y')
  f    = symbols('f', cls=Function)
  f    = 100*(y - x**2)**2 + (1 - x)**2
  J11  = diff(f,x)
  J12  = diff(f,y)
  J    = [J11,J12]
  H    = hessian(f, [x,y])
  return f,J,H,x,y

def line_search( f, J, H, x1,x2, X, rho, c, method=0):
  a_k = 1
  x_k = X

  # f( x_k + a p_k )
  p_k, g = pk(J, H, x1, x2, X, method)
  x_inner = (x_k+a_k*p_k)
  f_lhs = evalu(f,x1,x2,x_inner)
  
  # c a grad( f_k )T p_k
  f_xk  = evalu(f,x1,x2,x_k)
  f_rhs = f_xk + c*a_k*np.dot( np.transpose(g), p_k )

  while (f_lhs > f_rhs):
    a_k = rho*a_k
   
    x_inner = (x_k+a_k*p_k)
    f_lhs = evalu(f,x1,x2,x_inner)
    f_xk  = evalu(f,x1,x2,x_k)
    f_rhs = f_xk + c*a_k*np.dot( np.transpose(g), p_k )
  return a_k

def grad_desc( f, X, J, H, x1,x2, err ):
  x_points = []
  y_points = []
  x_min = X
  x_points.append(X.item(0))
  y_points.append(X.item(1))
  p_k,g = pk(J, H, x1, x2, x_min, 0)
  a_k = line_search( f, J, H, x1,x2,x_min, 0.9, 0.7, 0)

  while abs( LA.norm(a_k*g) ) > err:
      print a_k
      p_k,g = pk(J, H, x1, x2, x_min, 0)
      a_k = line_search( f, J, H, x1,x2,x_min, 0.9, 0.7, 0)
      x_min -= + a_k*g
      x_points.append(X.item(0))
      y_points.append(X.item(1))
  return x_min,x_points,y_points

def newton(f, J, H, x1, x2, X, err):
  x_points = []
  y_points = []
  x_min = X 
  x_points.append(X.item(0))
  y_points.append(X.item(1))
  step, g = pk(J, H, x1, x2, x_min, 1)
  L       = np.dot( np.transpose(g), -1*step )
  while (L/2. > err):
    a_k     = line_search( f, J, H, x1,x2,x_min, 0.9, 0.7, 1)
    print a_k
    step, g = pk(J, H, x1, x2, x_min, 1)
    L       = np.dot( np.transpose(g), -1*step )
    x_min  += a_k*step
    x_points.append(X.item(0))
    y_points.append(X.item(1))
  return x_min, x_points, y_points

def pk(J, H, x1, x2, X, method):
  g11 = J[0].subs([(x1,X[0]),(x2,X[1])])
  g21 = J[1].subs([(x1,X[0]),(x2,X[1])])

  g  = np.empty([2,1],dtype=float)
  g[0][0] = g11
  g[1][0] = g21

  if (method == 0):
    Bk = np.identity(2)
  else:
    B = H.subs([(x1,X[0]),(x2,X[1])])
    Bk = np.zeros((2,2))
    Bk[0][0] = B[0]
    Bk[0][1] = B[1]
    Bk[1][0] = B[2]
    Bk[1][1] = B[3]
  return np.dot(-1*np.linalg.inv(Bk),g), g

def evalu(f,x1,x2,X):
  return f.subs([(x1,X[0][0]),(x2,X[1][0])])

# PLOTTING

def rosen(x):
    return sum(100*(x[1:]-x[:-1]**2.0)**2.0 +(1-x[:-1])**2.0)

if __name__ == "__main__":
  f,J,H,x1,x2 = rosenbrock()
  X             = np.transpose( np.matrix ([[1.2,1.2]]))
  #X             = np.transpose( np.matrix ([[-1.2,1.]]))
  err           = 0.0001

  x_minimizer, x_points, y_points = grad_desc(f,X,J,H,x1,x2,err)
  #x_minimizer, x_points, y_points = newton(f, J, H, x1, x2, X, err)

  # Plotting contours
  xlist = np.linspace(-3,3,100)
  ylist = np.linspace(-3,3,100)

  X,Y = np.meshgrid(xlist,ylist)

  Z = rosen(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))

  plt.figure() 
  plt.contour(X,Y,Z, np.arange(10)**5)

  # Plotting path of algorithm
  plt.plot(x_points, y_points, '-o')
  plt.show()
