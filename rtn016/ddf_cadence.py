"""Create sample DDF cadence plots"""

# imports
import sys
import logging

import sqlite3
import healpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants

#from . import plotprep
import plotprep

# constants

SIM_DATABASE_FNAME = "/data/des91.b/data/neilsen/LSST/devel/sim-data/sims_featureScheduler_runs1.7/baseline/baseline_nexp2_v1.7_10yrs.db"
MJD_MIN = 60175
MJD_MAX = 60310
FIGURE_FNAME = "figures/ddf_cadence.pdf"

# exception classes

# interface functions


def main():
    """Make the sample cadence figure.
    """
    plotprep.config_logging()
    plotprep.config_matplotlib()

    logging.info("Reading %s", SIM_DATABASE_FNAME)
    with sqlite3.connect(SIM_DATABASE_FNAME) as con:
        visits = pd.read_sql_query("SELECT * FROM SummaryAllProps", con)

    logging.info("Making plot")
    fig, axes = plot_ddf_cadences(visits, MJD_MIN, MJD_MAX)
    logging.info("Saving plot")
    fig.savefig(FIGURE_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)
    logging.info("GLORIOUS SUCCESS")
    return 0


def find_field_visits(visits, ra, decl, nside=512, field_radius_deg=1.75):
    """Return visits centered near a pointing

    Parameters
    ----------
    visits : `pandas.DataFrame`
        The visits in which to look for fields
    ra : `float`
        The RA around which to search.
    decl : `float`
        The declination around which to search
    nside : `int`
        The nside for the healpix search
    field_radious_deg : `float`
        The radious round which to search, in degrees

    Returns
    -------
    field_visits : `pandas.DataFrame`
        The visits on the field.
    """
    field_hpxs = healpy.query_disc(
        nside,
        healpy.ang2vec(ra, decl, lonlat=True),
        np.radians(field_radius_deg),
    )
    visit_hpxs = healpy.ang2pix(
        nside, visits["fieldRA"].values, visits["fieldDec"].values, lonlat=True
    )
    field_visits = visits.loc[np.isin(visit_hpxs, field_hpxs)]
    return field_visits


def plot_field_cadence(visits, mjd_range=None, ax=None):
    """Plot the cadence for a field.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        The visits on the field
    ax : `matplotlib.axes.Axes`
        The axes on which to plot the cadence

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The axes on which the cadence was plotted
    """
    # Add 0.2 because sometimes a night at Cerro Pachon starts just before the
    # MJD rollover.
    night_mjd = np.floor(0.2 + visits.observationStartMJD).astype(int)
    nights = (
        visits.assign(
            teff=plotprep.compute_teff(
                visits.fiveSigmaDepth, visits["filter"]
            ),
            night_mjd=night_mjd,
        )
        .loc[:, ["filter", "night_mjd", "teff"]]
        .groupby(["filter", "night_mjd"])
        .sum()
        .sort_index()
    )

    if mjd_range is not None:
        night_mjds = np.arange(*mjd_range)
    else:
        night_mjds = np.arange(night_mjd.min(), night_mjd.max() + 1)

    if ax is None:
        fig, ax = plt.subplots()

    accum_teff = None
    for band in plotprep.BANDS:
        if band not in visits["filter"].values:
            continue

        # Need to reindex so accum_teff has the same shape across all bands
        band_nights = nights.loc[band].reindex(night_mjds, fill_value=0)

        ax.bar(
            band_nights.index,
            band_nights.teff,
            bottom=accum_teff,
            width=1,
            color=plotprep.BAND_COLOR[band],
            label=band,
        )
        if accum_teff is None:
            accum_teff = band_nights.teff
        else:
            accum_teff += band_nights.teff

    return ax


def plot_ddf_cadences(visits, mjd_min, mjd_max):
    """Full cadence plot for all DDFs

    Parameters
    ----------
    visits : `pandas.DataFrame`
        the visits
    mjd_min : `int`
        the first MJD to plot
    mjd_max : `int`
        the last MJD to plot

    Returns
    -------
    fig : the `matplotlib.figure.Figure`
        the figure with the plot
    axes : `list(matplotlib.axes.Axes)`
        the axes in the plot
    """
    fig, axes = plt.subplots(
        len(plotprep.DDF_FIELDS),
        sharex=True,
        sharey=True,
        figsize=(11 * scipy.constants.golden, 11),
        gridspec_kw={"hspace": 0.05},
    )
    for field, ax in zip(plotprep.DDF_FIELDS.index, axes):
        field_visits = find_field_visits(
            visits,
            plotprep.DDF_FIELDS.loc[field, "ra"],
            plotprep.DDF_FIELDS.loc[field, "decl"],
        )
        plot_field_cadence(field_visits, mjd_range=(mjd_min, mjd_max), ax=ax)
        ax.annotate(field, xy=(0.02, 0.85), xycoords="axes fraction")

    axes[0].legend()
    axes[-1].set_xlabel("date")
    axes[2].set_ylabel("$t_{eff}$")

    start_date = pd.to_datetime(mjd_min + 2400000.5, unit="D", origin="julian")
    end_date = pd.to_datetime(mjd_max + 2400000.5, unit="D", origin="julian")
    date_seq = pd.date_range(start=start_date, end=end_date, freq="MS")
    axes[-1].set_xticks(date_seq.to_julian_date() - 2400000.5)
    axes[-1].set_xticklabels(str(d)[:10] for d in date_seq)
    return fig, axes


# classes

# internal functions & classes

if __name__ == "__main__":
    sys.exit(main())
