import numpy as np
from pyquaternion import Quaternion


def quaternion_yaw(q: Quaternion) -> float:
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])
    return yaw
