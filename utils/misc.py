"""
Misc utilities

"""
import pandas as pd
import numpy as np


def dataloader(path, dia, height, idx_load_strt, idx_load_end):
    """
    Dataloader to load E-mod tensile test data.

    :param dia :  DIameter of the specimen in m
    :param height : Height of the specimen in mm
    idx_load_strt : Initial rows to skip to reach to a point where load is starting
    idx_load_end : Last rows to skip
    """
    data = pd.read_csv(path, delimiter="\t", skiprows=14, skipfooter=idx_load_end)

    # Converting , to . so that import the values as float rather than string
    data['mm.1'] = [x.replace(',', '.') for x in data['mm.1']]

    data['mm.1'] = data['mm.1'].astype(float)
    data['mm.2'] = [x.replace(',', '.') for x in data['mm.2']]

    data['mm.2'] = data['mm.2'].astype(float)
    data['mm.3'] = [x.replace(',', '.') for x in data['mm.3']]

    data['mm.3'] = data['mm.3'].astype(float)
    data['s'] = [x.replace(',', '.') for x in data['s']]

    data['s'] = data['s'].astype(float)
    data['kN.1'] = [x.replace(',', '.') for x in data['kN.1']]

    data['kN.1'] = data['kN.1'].astype(float)

    # dropping initial load cycles
    data = data.drop(labels=range(0, idx_load_strt), axis=0)

    # creating stress and strain columns
    data['strain'] = ((data['mm.1'] + data['mm.2'] + data['mm.3']) / 3) / height
    data['stress'] = (data['kN.1'] * 1000) / (np.pi * (dia / 2) ** 2)

    return data
