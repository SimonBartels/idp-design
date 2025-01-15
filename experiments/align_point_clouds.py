import numpy as np


class PointCloudAlignment:
    def __init__(self, cloud1):
        self.P1 = cloud1
        self.C1 = np.sum(self.P1, axis=0) / self.P1.shape[0]
        self.P1 = self.P1 - self.C1

    def align(self, matching_points, point_to_align):
        assert(matching_points.shape[0] == self.P1.shape[0] and matching_points.shape[1] == self.P1.shape[1])
        P2 = matching_points
        C2 = np.sum(P2, axis=0) / P2.shape[0]
        P2 = P2 - C2

        M = P2.T @ self.P1
        U, _, VT = np.linalg.svd(M)
        Q = U @ VT

        return (point_to_align - C2) @ Q + self.C1
