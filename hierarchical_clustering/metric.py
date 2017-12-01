import numpy as np

#compute f-cja

#zawiera slownik i obsluguje
#rozme klucze ta sama wartosc
#Hierachical Clustering

class MetricsFunctions():
    def __init__(self,string):
        self.string = string
        self.my_dict = {'eucl': eucl, 'l1': l1, 'l2': l2}
    def compute(self,X,Y):
            if self.string in self.my_dict.keys():
                self.my_dict[self.string](X,Y)
            else:
                raise Exception('Bad Metric')





def eucl(X,Y):
    return np.sqrt(l2(X,Y))
def l1(X,Y):
    return np.sum(np.abs(X-Y))
def l2(X,Y):
    return np.sum(np.square((X-Y)))



<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
