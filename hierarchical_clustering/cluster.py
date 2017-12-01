import numpy as np
import hierarchical_clustering


class Cluster:
    def __init__(self, initial_point_id, step_info):
        self.points_id = np.array([initial_point_id])
        self.step_info = step_info

    def merge(self, other_id):
        pass

    def distance(self, self_id, other_id):
        pass


class ClusterMax(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)


class ClusterAverage(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)


class ClusterWard(Cluster):
    def __init__(self, initial_point_id, step_info):
        Cluster.__init__(self, initial_point_id, step_info)


if __name__ == "__main__":
    step_info_ = hierarchical_clustering.StepInfo()
    step_info_.current_distance = np.array([1, 2])
    cluster = ClusterMax(2, step_info_)
    step_info_.current_distance = np.array([7, 8, 9])
    print(cluster.step_info.current_distance)
