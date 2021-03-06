import numpy as np

#compute f-cja

#zawiera slownik i obsluguje
#rozme klucze ta sama wartosc
#Hierachical Clustering


class MetricsFunctions:
    def __init__(self, string):
        """
        Constructor of MetricsFuntions class
        :param string: name of metric
        """
        self.string = string
        self.my_dict = {'eucl': eucl, 'l1': l1, "l2": eucl, 'euclidean': eucl, 'manhattan': l1}

    def compute(self, x, y):
        """
        Functions that computes distance between set, metric depends on this class string parameter
        :param x: vector of first set of points
        :param y: vector of second set of points 
        :return: calculated metric value
        """
        x = np.array(x)
        y = np.array(y)
        if self.string in self.my_dict.keys():
            return self.my_dict[self.string](x, y)
        else:
            raise Exception('Bad Metric')


def eucl(x, y):
    """
    metric that computes distance as sum of square differences
    :param x: vector of first set of points 
    :param y: vector of second set of points 
    :return: calculated metric value
    """
    return np.sqrt(np.sum(np.square((x-y))))


def l1(x, y):
    """
     metric that computes distance as absolute value of differences
    :param x: vector of first set of points 
    :param y: vector of second set of points 
    :return: calculated metric value
    """
    return np.sum(np.abs(x-y))



