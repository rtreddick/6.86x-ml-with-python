import numpy as np

def perceptron(points, labels, theta=None, theta_0=0, start_index=0):
    '''
    points is list of points
    label is list of ints -1 or 1
    theta is a vector of ints
    theta_0 is an int
    start_index sets the starting point for iterating through points
    '''

    if theta == None:
        theta = np.zeros(len(points[0]))
    else:
        theta = np.array(theta)

    items = [*zip(points, labels)]
    items = [*items[start_index:], *items[:start_index]] if start_index else items
    thetas = [theta]
    thetas_0 = [theta_0]
    mistakes = {point: 0 for point in points}
    mistake_point = 0
    
    # iterate through points until success
    while True:

        for index, (point, label) in enumerate(items):

            if label*np.matmul(theta.transpose(), np.array(point)) <= 0:
                theta = theta + label*np.array(point)
                theta_0 = theta_0 + label
                thetas.append(theta)
                thetas_0.append(theta_0)
                mistake_point = index
                mistakes[point] = mistakes.get(point, 0) + 1
            
            # if return to the same point with no more mistakes, success
            elif index == mistake_point:
                return thetas, thetas_0, sum(mistakes.values()), mistakes

start_index = 0
points = [(-4,2), (-2,1), (-1,-1), (2,2), (1,-2)]
labels = [1, 1, -1, -1, -1]
result = perceptron(points, labels, theta=[-1,1], theta_0=-2, start_index=start_index)

print(f"thetas: {result[0]}")
print(f"thetas_0: {result[1]}")
print(f"total mistakes: {result[2]}")
print(f"mistakes by point:")
for key, val in result[3].items():
    print(f"{key}: {val}")