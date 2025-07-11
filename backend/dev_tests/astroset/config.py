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
