import numpy as np

#compute f-cja

#zawiera slownik i obsluguje
#rozme klucze ta sama wartosc
#Hierachical Clustering


class MetricsFunctions:
    def __init__(self, string):
        self.string = string
        self.my_dict = {'eucl': eucl, 'l1': l1, 'l2': l2}

    def compute(self, x, y):
        x = np.array(x)
        y = np.array(y)
        if self.string in self.my_dict.keys():
            return self.my_dict[self.string](x, y)
        else:
            raise Exception('Bad Metric')


def eucl(x, y):
    return np.sqrt(l2(x, y))


def l1(x, y):
    return np.sum(np.abs(x-y))


def l2(x, y):
    return np.sum(np.square((x-y)))

"""
metric = MetricsFunctions("eucl")
left_list = [9, 1]
left_list = np.asarray(left_list)
right_list = [2, 3]
right_list = np.asarray(right_list)
print(metric.compute(left_list, right_list))
"""
"""
a = [[1,2,0],[1,1,1],[3,3,3],[4,4,4]]
a = np.array(a)
print(a.argmin(axis=0))
print(a.argmin(axis=1))
print(np.min(np.min(a, axis=1), axis=0))
print(np.argmin(a,axis=0))
print(np.argmin(a,axis=1))

print(np.amin(a[:,1]))
"""