# Target Function
Rosenbrock: f(x) = 100(x2-x1^2)^2 + (1-x1)^2

# Initial guess
The initial point is either (1.2, 1.2) or (-1.2, 1).
To start at a different initial guess, change lines 105/106.

# Minimization Algorithm
Uses either Newton's Method or Gradient Descent. 

Both of these algorithms will return three values:
  1) a vector representing the minimizer
  2) x_points: an array of the x1 coordinates of the points taken 
              by the algorithm at each iteration
  3) y_points: an array of the x2 coordinates of the points taken 
              by the algorithm at each iteration

# Step Size Selection
Backtracking line search with initial α = 1. 

### Step Size Values

The file a_k.csv contains the a_k values of the algorithms through
every iteration. The first column is for the newton's method when
the initial point is (-1.2,1.0). The second column is newton's method
when initial point is (1.2,1.2). The third column is gradient descent
when starting at (1.2,1.2) and the final column is gradient descent
when starting at point (-1.2,1.0).

# Plots
The file grad_desc_x_-1.2_y_1.png corresponds to gradient descent
  when the initial starting point is (-1.2,1)
The file grad_desc_x_1.2_y_1.2.png corresponds to gradient descent
  when the initial starting point is (1.2,1.2)
The file newton_x_-1.2_y_1.png corresponds to newtons method
  when the initial starting point is (-1.2,1)
The file newton_x_1.2_y_1.2.png corresponds to newtons method
  when the initial starting point is (1.2,1.2)
