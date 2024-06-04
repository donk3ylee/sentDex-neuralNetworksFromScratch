import numpy as np

# Now using a batch of inputs (list of lists) rather than a vector list
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Because we are now using a batch of inputs we need to transpose the weights
# so that we dont get a shape error. Note we swapped the dot order too.
output = np.dot(inputs, np.array(weights).T) + biases
print(output)

# UPTO pt.4 17.05