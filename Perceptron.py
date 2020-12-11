import numpy as np

class Perceptron(object):
    def __init__(self, step_size=0.01, num_iter=50, random_state=1):
        self.step_size = step_size #Learning rate, between 0.0 and 1.0
        self.num_iter = num_iter #Number of passes over the training dataset
        self.random_state = random_state #RNG seed for random weight initialization
    
    def fit(self, X, y):
        #weight_array is a 1d-array of weights after fitting
        #errors_list is a list detailing numbers of misclassifications (i.e. updates) in each epoch (i.e. iteration).
        rgen = np.random.RandomState(self.random_state)
        self.weight_array = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_list = []
        
        for num in range(0, self.num_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.step_size * (target - self.predict(xi))
                self.weight_array[1:] += update * xi
                self.weight_array[0] += update
                errors += int(update != 0.0)
            self.errors_list.append(errors)
        return self
    
    def net_input(self, X):
        #Gives dot product of two arrays
        return np.dot(X, self.weight_array[1:]) + self.weight_array[0]
    
    def predict(self, X):
        #where net_input(X) is greater than or equal to 0 (threshold), classify as 1, else -1
        return np.where(self.net_input(X) >= 0.0, 1, -1)