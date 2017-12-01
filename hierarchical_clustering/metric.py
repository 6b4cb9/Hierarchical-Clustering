import numpy as np

#compute f-cja

#zawiera slownik i obsluguje
#rozme klucze ta sama wartosc
#Hierachical Clustering


class MetricsFunctions:
    def __init__(self, string):
        """
        :param string: name of metric
        """
        self.string = string
        self.my_dict = {'eucl': eucl, 'l1': l1, 'l2': l2}

    def compute(self, x, y):
        """
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
    
    :param x: vector of first set of points 
    :param y: vector of second set of points 
    :return: calculated metric value
    """
    return np.sqrt(np.sum(np.square((x-y))))


def l1(x, y):
    """
    
    :param x: vector of first set of points 
    :param y: vector of second set of points 
    :return: calculated metric value
    """
    return np.sum(np.abs(x-y))


"""
metric = MetricsFunctions("eucl")
left_list = [9, 1]
left_list = np.asarray(left_list)
right_list = [2, 3]
right_list = np.asarray(right_list)
print(metric.compute(left_list, right_list))
"""

a = [[1,2,1],[1,1,0],[3,3,3],[4,4,4]]
a = np.array(a)
print(np.argmin(a))
print(a.argmin(axis=0))
print(a.argmin(axis=1))
print(np.min(np.min(a, axis=1), axis=0))
print(np.argmin(a,axis=0))
print(np.argmin(a,axis=1))

def coordinates(array):

    #x=np.mod(np.argmin(array),np.shape(array)[0])
    #x=np.mod(np.argmin(array),np.shape(array)[0])
    #y=np.argmin(array)//array[0].size
    p,q=np.unravel_index(array.argmin(), array.shape)

    return np.unravel_index(array.argmin(), array.shape)

print(coordinates(a))


#print(np.amin(a[:,1]))
