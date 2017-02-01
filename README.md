# Initial guess

If line 105 is uncommented, the initial point will be (1.2, 1.2).
If line 106 is uncommented, the initial point will be (-1.2, 1).
To start at a different initial guess, follow the structure of lines 105/106.

# Minimizaiton Algorithm

To run gradient descent, uncomment line 109.
To run newton's method, uncomment line 110.
 
Both of these algorithms will return three values:
  1) a vector representing the minimizer
  2) x_points: an array of the x1 coordinates of the points taken 
              by the algorithm at each iteration
  3) y_points: an array of the x2 coordinates of the points taken 
              by the algorithm at each iteration

# Plotting

Everything below line 112 simply plots the graph.

# Alpha Values

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
