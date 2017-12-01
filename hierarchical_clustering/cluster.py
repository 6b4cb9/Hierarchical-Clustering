import numpy as np
import hierarchical_clustering


class Cluster:
    def __init__(self, initial_point_id, step_info):
        if type(step_info) is not hierarchical_clustering.StepInfo:
            raise Exception("argument step_info must be class StepInfo")
        self.points_id = np.array([initial_point_id])
        self.step_info = step_info

    def __str__(self):
        return str(self.points_id)

    def merge(self, other_id):
        other = self.step_info.cluster_list[other_id]
        self.points_id = np.append(self.points_id, other.points_id)

    def distance(self, self_id, other_id):
        pass


class ClusterMax(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)

    def distance(self, self_id, other_id):
        other = self.step_info.cluster_list[other_id]
        distances = np.copy(self.step_info.initial_distance)

        distances = distances[self.points_id]
        distances = distances.transpose()
        distances = distances[other.points_id]
        distances = distances.transpose()

        return np.max(distances)

class ClusterAverage(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)

    def distance(self, self_id, other_id):
        other = self.step_info.cluster_list[other_id]
        distances = np.copy(self.step_info.initial_distance)

        distances = distances[self.points_id]
        distances = distances.transpose()
        distances = distances[other.points_id]
        distances = distances.transpose()

        return np.average(distances)


class ClusterWard(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)


if __name__ == "__main__":
    step_info_ = hierarchical_clustering.StepInfo()
    step_info_.initial_distance = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]]) / 10
    step_info_.current_distance = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]])

    cluster0 = ClusterAverage(0, step_info_)
    cluster1 = ClusterAverage(1, step_info_)
    cluster2 = ClusterAverage(2, step_info_)

    step_info_.cluster_list = np.array([cluster0, cluster1, cluster2])

    cluster0.merge(1)

    print(cluster0, cluster1, cluster2)
    print(cluster0.distance(0, 1))


