import numpy as np


def custom_distance(positions, cell_lengths, center=None):
    '''
    Calculate the distance matrix with the respect to periodic boundary conditions XYZ.
    :param positions: np.array. set of coordinates
    :param cell_lengths: simulation box size
    :param center: if the distance to point is necessary
    :return: distance matrix
    '''
    cell_lengths = np.array(cell_lengths)

    if center is None:
        distance = np.abs(positions[:, None, :] - positions)
    else:
        distance = np.abs(positions[:, None, :] - center)

    distance = np.where(distance > 0.5 * cell_lengths, distance - cell_lengths, distance)
    return np.sqrt((distance ** 2).sum(axis=-1))

def custom_distance_without_Z_pbc(positions, cell_lengths, center=None):
    '''
    Calculate the distance matrix with the respect to periodic boundary conditions XY.
    :param positions: np.array. set of coordinates
    :param cell_lengths: simulation box size
    :param center: if the distance to point is necessary
    :return: distance matrix
    '''
    cell_lengths = np.array(cell_lengths)

    if center is None:
        distance = np.abs(positions[:, None, :] - positions)
    else:
        distance = np.abs(positions[:, None, :] - center)

    distance[:, :, 0:-1] = np.where(distance[:, :, 0:-1] > 0.5 * cell_lengths[0:-1],
                                    distance[:, :, 0:-1] - cell_lengths[0:-1],
                                    distance[:, :, 0:-1])

    return np.sqrt((distance ** 2).sum(axis=-1))

def custom_distance_2d(positions, cell_lengths, center=None):
    '''
    Calculate the distance matrix with the respect to periodic boundary conditions XY.
    :param positions: np.array. set of coordinates
    :param cell_lengths: simulation box size
    :param center: if the distance to point is necessary
    :return: distance matrix
    '''
    positions = positions[:, 0:-1]
    cell_lengths = np.array(cell_lengths)
    cell_lengths = cell_lengths[0:-1]

    if center is None:
        distance = np.abs(positions[:, None, :] - positions)
    else:
        center = center[0:-1]
        distance = np.abs(positions[:, None, :] - center)

    distance = np.where(distance > 0.5 * cell_lengths,
                                    distance - cell_lengths,
                                    distance)

    return np.sqrt((distance ** 2).sum(axis=-1))

if __name__ == '__main__':
    print(custom_distance_without_Z_pbc(np.array([[5, 5, 1], [5, 5, 9]]), np.array([10, 10, 10]), center=None))
    print(custom_distance_without_Z_pbc(np.array([[1, 5, 5], [9, 5, 5]]), np.array([10, 10, 10]), center=None))
    print(custom_distance_without_Z_pbc(np.array([[5, 1, 5], [5, 9, 5]]), np.array([10, 10, 10]), center=None))
    print(custom_distance_2d(np.array([[5, 6, 1], [6, 5, 2]]), np.array([10, 10, 10]), center=np.array([5, 5, 0])))
    print(custom_distance_2d(np.array([[2, 1, 1], [9, 1, 2]]), np.array([10, 10, 10]), center=np.array([1, 1, 0])))
    pass
