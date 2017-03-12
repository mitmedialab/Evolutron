"""
Test of loading datasets.

Current datasets: type2p, b1h, m6a
"""
from evolutron.tools import load_dataset, file_db


# def test_load_dataset():
#     print('\nTesting dataset loading...')
#
#     for data_id in file_db.keys():
#         print('\nDataset: {} '.format(data_id))
#         x_data, y_data = load_dataset(data_id, padded=True, min_aa=None, max_aa=1000)
#         assert x_data
#
#         if y_data:
#             print(len(y_data))
#
#         del x_data, y_data
