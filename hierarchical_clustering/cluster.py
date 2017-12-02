import numpy as np


class StepInfo:
    """
    Class containing all important data for computing step in HierarchicalClustering including cluster class.
    """
    def __init__(self):
        """
        Constructor.
        """
        self.cluster_list = None
        self.initial_distance = None
        self.current_distance = None
        self.cluster_class = None

    def select_class(self, name):
        """
        Select cluster class.
        :param name: name of cluster class; possibilities = "complete", "max", "average", "ward"
        :return:
        """
        possibilities = {"complete": ClusterMax, "max": ClusterMax,
                         "average": ClusterAverage, "ward": ClusterWard}
        if name in possibilities.keys():
            self.cluster_class = possibilities[name]
        else:
            raise Exception('Bad cluster class')


class Cluster:
    """
    Interface of cluster.
    """
    step_info = StepInfo()

    def __init__(self, initial_point_id):
        """
        Constructor of Cluster class.
        :param initial_point_id: id of first point in cluster
        :param step_info: reference to object of class StepInfo
        """
        self.points_id = np.array([initial_point_id])

    def __str__(self):
        """
        Function describing what to show while using print function.
        :return: self.points_id as string
        """
        return str(self.points_id)

    def merge(self, other_id):
        """
        Merge a cluster with this one.
        :param other_id: index of cluster in self.step_info.cluster_list which is to merge with this cluster
        :return:
        """
        other = self.step_info.cluster_list[other_id]
        self.points_id = np.append(self.points_id, other.points_id)

    @staticmethod
    def distance(self_id, other_id):
        """
        Virtual function to compute distance between this cluster and another.
        :param self_id: index of this cluster in self.step_info.cluster_list
        :param other_id: index of the other cluster in self.step_info.cluster_list
        :return:
        """
        pass


class ClusterMax(Cluster):
    """
    Cluster class for max linkage.
    """
    def __init__(self, initial_point_id):
        """
        :param initial_point_id: id of first point in cluster
        :param step_info: reference to object of class StepInfo
        """
        Cluster.__init__(self, initial_point_id)

    @staticmethod
    def distance(self_id, other_id):
        """
        Function to compute distance between this cluster and another using max linkage.
        :param self_id: index of this cluster in self.step_info.cluster_list
        :param other_id: index of the other cluster in self.step_info.cluster_list
        :return: the distance
        """
        self = Cluster.step_info.cluster_list[self_id]
        other = Cluster.step_info.cluster_list[other_id]
        distances = np.copy(Cluster.step_info.initial_distance)

        distances = distances[self.points_id]
        distances = distances.transpose()
        distances = distances[other.points_id]
        distances = distances.transpose()

        return np.max(distances)


class ClusterAverage(Cluster):
    """
    Cluster class for average linkage.
    """
    def __init__(self, initial_point_id):
        """
        :param initial_point_id: id of first point in cluster
        :param step_info: reference to object of class StepInfo
        """
        Cluster.__init__(self, initial_point_id)

    @staticmethod
    def distance(self_id, other_id):
        """
        Function to compute distance between this cluster and another using average linkage.
        :param self_id: index of this cluster in self.step_info.cluster_list
        :param other_id: index of the other cluster in self.step_info.cluster_list
        :return: the distance
        """
        self = Cluster.step_info.cluster_list[self_id]
        other = Cluster.step_info.cluster_list[other_id]
        distances = np.copy(Cluster.step_info.initial_distance)

        distances = distances[self.points_id]
        distances = distances.transpose()
        distances = distances[other.points_id]
        distances = distances.transpose()

        return np.average(distances)


class ClusterWard(Cluster):
    """
    Cluster class for ward linkage.
    """
    def __init__(self, initial_point_id):
        """
        :param initial_point_id: id of first point in cluster
        :param step_info: reference to object of class StepInfo
        """
        Cluster.__init__(self, initial_point_id)
        self._merged_id = initial_point_id
        self._old_points_size = 1

    def merge(self, other_id):
        """
        Merge a cluster with this one.
        :param other_id: index of cluster in self.step_info.cluster_list which is to merge with this cluster
        :return:
        """
        self._old_points_size = self.points_id.size
        other = self.step_info.cluster_list[other_id]
        self.points_id = np.append(self.points_id, other.points_id)
        self._merged_id = other_id

    @staticmethod
    def distance(self_id, other_id):
        """
        Function to compute distance between this cluster and another using ward linkage.
        :param self_id: index of this cluster in self.step_info.cluster_list
        :param other_id: index of the other cluster in self.step_info.cluster_list
        :return: the distance
        """
        self = Cluster.step_info.cluster_list[self_id]
        distances = np.copy(Cluster.step_info.current_distance)
        merged_id = self._merged_id

        if merged_id == self_id:
            #if used before merging
            return distances[self_id, other_id]
        else:
            self_size = self._old_points_size
            merged_size = Cluster.step_info.cluster_list[merged_id].points_id.size
            other_size = Cluster.step_info.cluster_list[other_id].points_id.size

            denominator = self_size + merged_size + other_size
            a1 = (self_size + other_size)/denominator
            a2 = (merged_size + other_size)/denominator
            b = -other_size/denominator

            return a1*distances[self_id, other_id] + a2*distances[merged_id, other_id] + b*distances[self_id, merged_id]



if __name__ == "__main__":
    step_info_ = Cluster.step_info
    step_info_.select_class("ward")
    step_info_.initial_distance = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]]) / 10
    step_info_.current_distance = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])

    cluster0 = step_info_.cluster_class(0)
    cluster1 = step_info_.cluster_class(1)
    cluster2 = step_info_.cluster_class(2)

    step_info_.cluster_list = np.array([cluster0, cluster1, cluster2])

    cluster0.merge(1)

    print(cluster0, cluster1, cluster2)
    print(step_info_.cluster_class.distance(0, 1))


