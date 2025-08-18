"""
config.py
---------
Configuration for Astrodash-style training set creation pipeline.
Returns a Python dict with all necessary parameters.
"""

def get_config():
    """
    Returns the configuration parameters for the training set pipeline.
    Modify this function to change dataset parameters.
    """
    astrodash_to_transformer_map = {
        'Ia-norm': 'Ia',
        'Ia-91T': 'Ia',
        'Ia-91bg': 'Ia',
        'Ia-csm': 'Ia',
        'Iax': 'Ia',
        'Ia-pec': 'Ia',
        'Ib-norm': 'Ib/c',
        'Ibn': 'Ib/c',
        'IIb': 'II',
        'Ib-pec': 'Ib/c',
        'Ic-norm': 'Ib/c',
        'Ic-broad': 'Ib/c',
        'Ic-pec': 'Ib/c',
        'IIP': 'II',
        'IIL': 'II',
        'IIn': 'IIn',
        'II-pec': 'II', # Doesn't have any SLSNe-I
    }
    transformer_class_to_idx = {
        'Ia': 0,
        'IIn': 1,
        'SLSNe-I': 2,
        'II': 3,
        'Ib/c': 4,
        'IIP': 5,
        'IIL': 6,
    }

    parameters = {
        'typeList': [
            'Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Iax', 'Ia-pec',
            'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
            'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec'
        ],
        'nTypes': 17,
        'w0': 3500.,  # wavelength range in Angstroms
        'w1': 10000.,
        'nw': 1024,   # number of wavelength bins
        'minAge': -20.,
        'maxAge': 50.,
        'ageBinSize': 4.,
        'galTypeList': [
            'E', 'S0', 'Sa', 'Sb', 'Sc', 'SB1', 'SB2', 'SB3', 'SB4', 'SB5', 'SB6'
        ]
    }
    return parameters
