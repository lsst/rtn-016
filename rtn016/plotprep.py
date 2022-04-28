"""Common code for sample plots"""

# imports
import random
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants


# constants

# DESC standard
BAND_COLOR = {
    "u": "#56b4e9",
    "g": "#008060",
    "r": "#ff4000",
    "i": "#850000",
    "z": "#6600cc",
    "y": "#000000",
}

BANDS = ('u', 'g', 'r', 'i', 'z', 'y')
M0 = pd.Series({'u': 23.9, 'g': 25.0, 'r': 24.7, 'i': 24.0, 'z': 23.3, 'y': 22.1})
M0.index.name = 'filter'

NUMPY_RANDOM_SEED = 6563
STDLIB_RANDOM_SEED = 4861

DDF_FIELDS = pd.DataFrame(
    [
        {"ra": 9.45, "decl": -44.0, "field_name": "Elias S1"},
        {"ra": 35.708333, "decl": -4.75, "field_name": "XMM-LSS"},
        {"ra": 53.125, "decl": -28.1, "field_name": "ECDFS"},
        {"ra": 58.97, "decl": -49.28, "field_name": "Euclid 1"},
        {"ra": 63.6, "decl": -47.60, "field_name": "Euclid 2"},
        {"ra": 150.1, "decl": 2.1819444, "field_name": "COSMOS",},
    ]
).set_index("field_name")

# exception classes

# interface functions


def config_logging():
    """Configure logging for plot creation.
    """
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)


def config_matplotlib():
    """Configure matplotlib for sample plots.
    """
    width = 7.5
    height = scipy.constants.golden * width
    mpl.rcParams["figure.figsize"] = (height, width)
    plt.style.use("ggplot")


def seed_random():
    """Set standard random number seeds.
    """
    random.seed(STDLIB_RANDOM_SEED)
    np.random.seed(NUMPY_RANDOM_SEED)


def config():
    """Configure for sample plots.
    """
    config_logging()
    config_matplotlib()
    seed_random()

def compute_teff(mag, band, exptime=30):
    """Compute t_eff for a given magnitude in a given band

    Parameters
    ----------
    mag : the 5-sigma limiting magnitudes
    band : the bands
    exptime : the exposure time in seconds

    Returns:
    teff : the effective exposure time
    """
    teff = exptime * 10**(0.8*(mag - band.map(M0)))
    return teff

# classes

# internal functions & classes
