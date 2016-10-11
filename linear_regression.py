import sklearn.linear_model

from core import Solver


class LinearRegression(Solver):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.solver = sklearn.linear_model.LinearRegression()


if __name__ == '__main__':
    linreg = LinearRegression()
    linreg.train()
    linreg.solve()
