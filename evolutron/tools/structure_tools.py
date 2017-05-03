import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import time


elements_map = {'H': 0,
                'C': 1,
                'N': 2,
                'O': 3,
                'S': 4,
                'X': 5}


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


def rotatePDBdf(df, angles):
    # 3D rotation of the x,y,z coordinates in a dataframe

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Rotation matrix
    R = np.array([[cos[1]*cos[2], cos[2]*sin[0]*sin[1]-cos[0]*sin[2], cos[0]*cos[2]*sin[1]+sin[0]*sin[2]],
                  [cos[1]*sin[2], cos[0]*cos[2]+sin[0]*sin[1]*sin[2], -cos[2]*sin[0]+cos[0]*sin[1]*sin[2]],
                  [-sin[1], cos[1]*sin[0], cos[0]*cos[1]]])

    M = np.array([df.x_coord.values, df.y_coord.values, df.z_coord.values])

    rotM = np.matmul(R, M)

    df.x_coord = rotM[0]
    df.y_coord = rotM[1]
    df.z_coord = rotM[2]

    return df
