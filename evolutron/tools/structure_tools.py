import numpy as np
from scipy.spatial.distance import pdist, squareform

elements_map = {'H': 0,
                'C': 1,
                'N': 2,
                'O': 3,
                'S': 4,
                'X': 5}


def one_hot_elements_map(x):
    arr = np.zeros(len(elements_map))
    arr[elements_map[x[0]]] = 1
    return arr


def PDBdf2npArray(atoms, res, size=np.Inf):
    num_channels = len(elements_map)

    x_coord = (((atoms.x_coord - atoms.x_coord.min()) / res).astype(int)).values
    y_coord = (((atoms.y_coord - atoms.y_coord.min()) / res).astype(int)).values
    z_coord = (((atoms.z_coord - atoms.z_coord.min()) / res).astype(int)).values
    channel_coord = np.vectorize(lambda x: elements_map[x])(atoms.element_symbol.values)

    max_size = max([x_coord.max(), y_coord.max(), z_coord.max()]) + 1

    if size >= max_size:
        max_size = size
        clip_flag = False
    else:
        clip_flag = True

    shape = (max_size, max_size, max_size, num_channels)

    arr = np.zeros(shape=shape, dtype=np.bool)
    arr[x_coord, y_coord, z_coord, channel_coord] = 1

    if clip_flag:
        arr = arr[:size, :size, :size, :]

    return arr, clip_flag


def PDBdf2linearArray(atoms, size=np.Inf):
    x_coord = (atoms.x_coord - atoms.x_coord.min()).values
    y_coord = (atoms.y_coord - atoms.y_coord.min()).values
    z_coord = (atoms.z_coord - atoms.z_coord.min()).values
    coords = np.stack([x_coord, y_coord, z_coord], axis=-1)

    atom_type = np.apply_along_axis(one_hot_elements_map, -1, np.expand_dims(atoms.element_symbol.values, axis=-1))
    # np.vectorize(one_hot_elements_map)(atoms.element_symbol.values)

    distances = squareform(pdist(coords))
    distances.partition(distances.shape[-1] - 4, axis=-1)
    dist = distances[:, -3:]

    arr = np.concatenate((coords, atom_type, dist), axis=-1)

    return arr


def rotatePDBdf(df, angles):
    # 3D rotation of the x,y,z coordinates in a dataframe

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Rotation matrix
    R = np.array(
            [[cos[1] * cos[2], cos[2] * sin[0] * sin[1] - cos[0] * sin[2], cos[0] * cos[2] * sin[1] + sin[0] * sin[2]],
             [cos[1] * sin[2], cos[0] * cos[2] + sin[0] * sin[1] * sin[2], -cos[2] * sin[0] + cos[0] * sin[1] * sin[2]],
             [-sin[1], cos[1] * sin[0], cos[0] * cos[1]]])

    M = np.array([df.x_coord.values, df.y_coord.values, df.z_coord.values])

    rotM = np.matmul(R, M)

    df.x_coord = rotM[0]
    df.y_coord = rotM[1]
    df.z_coord = rotM[2]

    return df


def rotateArray(arr, angles):
    # 3D rotation of the x,y,z coordinates in a dataframe

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Rotation matrix
    R = np.array(
            [[cos[1] * cos[2], cos[2] * sin[0] * sin[1] - cos[0] * sin[2], cos[0] * cos[2] * sin[1] + sin[0] * sin[2]],
             [cos[1] * sin[2], cos[0] * cos[2] + sin[0] * sin[1] * sin[2], -cos[2] * sin[0] + cos[0] * sin[1] * sin[2]],
             [-sin[1], cos[1] * sin[0], cos[0] * cos[1]]])

    M = arr[:, :3]

    rotM = np.matmul(R, M.transpose())

    rot_arr = np.concatenate((rotM.transpose(), arr[:, 3:]), axis=-1)

    return rot_arr
