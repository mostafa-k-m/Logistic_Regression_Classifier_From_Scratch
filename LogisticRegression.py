import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

class LogisticRegression:
    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.no_of_datapoints = max(train_X.shape)
        self.no_of_features = min(train_X.shape)
        self.cost_history = []
        self.W = np.zeros((self.no_of_features+1,1))
        self.classes = np.unique(train_y)
        self.checkpoints = {}


    def sigmoid(self,z):
        return(1/(1+np.exp(-1*z)))

    def compute_cost(self):
        epsilon =1e-5
        h = self.sigmoid(self.X_tr.dot(self.W))
        return (-1/self.no_of_datapoints) * (self.train_y.T.dot(np.log(h+epsilon))+ ((1-self.train_y).T.dot(np.log(1-h+epsilon)))), h

    def gradient_descent(self, learning_rate):
        cost, h = self.compute_cost()
        self.cost_history.append(cost)
        err = h-self.train_y
        grad = (1/self.no_of_datapoints)* (self.X_tr.T.dot(err))
        self.W=self.W-((learning_rate)*(grad))

    def train(self, iterations, learning_rate, two_d = True, draw_history = True, results = True):
        self.train_y = self.train_y.reshape(-1,1)
        self.X_tr=np.hstack((np.ones((self.no_of_datapoints,1)),self.train_X))
        for i in range(iterations):
            self.gradient_descent(learning_rate)
            checkpoints = [i for i in (iterations//4)*np.asarray(range(4))]
            if i in checkpoints:
                self.checkpoints[i] = [self.cost_history[-1], self.W]
        if results:
            print(f"@ no of iterations = {iterations} and learning rate = {learning_rate} \n final W = {self.W} \n final cost = {self.cost_history[-1][0]}")
            f = plt.figure(figsize = (12, 6))
            f.suptitle(f"Performance @ {iterations} iterations & learning rate = {learning_rate}", fontsize=16)
            if two_d:
                axes1 = f.add_subplot(1, 2, 1)
                axes2 = f.add_subplot(1,2,2)
                self.plot_decision_boundary(self.train_X, self.train_y.reshape(-1,), axes1, draw_history)
                self.plot_error_curve(axes2, two_d, draw_history)
            else:
                axes = f.add_subplot(1, 1, 1)
                self.plot_error_curve(axes, two_d, draw_history = False)
                return f

    def predict(self,test_X):
        no_of_test_datapoints = max(np.shape(test_X))
        test_X_tr=np.hstack((np.ones((no_of_test_datapoints,1)),test_X))
        y_pred=np.round(self.sigmoid(test_X_tr.dot(self.W)))
        return y_pred.reshape(-1,)

    def plot_decision_boundary(self, X, y, axes, draw_history):
        y = (y.astype(int)).astype(str)
        y = ["Class " + i for i in y]
        sns.scatterplot(X[:,0],X[:,1],hue = y, ax=axes)
        min, max = np.min(X[:, 0]), np.max(X[:, 0])
        x_values = [min - .05*min, max + .05*max]
        y_values = - (self.W[0] + (self.W[1]* x_values)) / self.W[2]
        sns.lineplot(x_values, y_values, color='k', label = "Final Decision Boundary", ax=axes)
        if draw_history:
            for i in self.checkpoints:
                y_value = (self.checkpoints[i][1][0] + (self.checkpoints[i][1][1]* x_values)) / self.checkpoints[i][1][2]
                sns.lineplot(x_values, y_value, label = f"Decision Boundary @ {i} iterations", alpha =.5, ax=axes)
        axes.set(title='Training Data & Decision Boundary')
        if draw_history:
            axes.legend(loc='lower center', bbox_to_anchor=(.5, -.47), ncol=1)
        else:
            axes.legend(loc='lower center', bbox_to_anchor=(.5, -.28), ncol=1)


    def plot_error_curve(self, axes, two_d, draw_history):
        cost = np.asarray(self.cost_history)[:,0,0]
        iterations = np.asarray(range(len(self.cost_history)))
        sns.lineplot(iterations, cost, color='k', label = "Error Curve", ax=axes)
        checkpoint_iterations = list(self.checkpoints.keys())
        checkpoint_costs = [i[0] for i in list(self.checkpoints.values())]
        if draw_history:
            sns.scatterplot(checkpoint_iterations, checkpoint_costs, color = 'r',s = 100, marker =  u'+', label = "Cost @ interval", ax = axes)
        axes.set(title='Error Curve', xlabel='No. of itetations', ylabel='Cost')
        if two_d:
            if draw_history:
                axes.legend(loc='lower center', bbox_to_anchor=(.5, -.24), ncol=1)
            else:
                axes.legend(loc='lower center', bbox_to_anchor=(.5, -.19), ncol=1)
