"""Create sample hourglass plots"""

# Start logging first so we can log imports
import logging

logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.info("Starting")

import os.path
import copy
import sys
from collections import OrderedDict

logger.debug("Loading common modules")
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants
import sqlite3
import yaml

logger.debug("Loading general astronomy modules")
import healpy
import palpy
import astropy
import astropy.coordinates
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u
from astropy.coordinates import Angle

logger.debug("Loading Rubin Observatory modules")
import lsst.sims.utils
from lsst.sims.utils import m5_flat_sed
from lsst.sims.seeingModel import SeeingData, SeeingModel
from lsst.sims.cloudModel import CloudData
from lsst.sims.skybrightness_pre import SkyModelPre


#from . import plotprep
import plotprep

# constants

SIM_DATABASE_FNAME = "/data/des91.b/data/neilsen/LSST/devel/sim-data/sims_featureScheduler_runs1.7/baseline/baseline_nexp2_v1.7_10yrs.db"
MONTH = 10
YEAR = 2024
MOON_PLOT_FNAME = "figures/moon_hourglass.png"
BLOCK_PLOT_FNAME = "figures/block_hourglass.png"
DEEP_COORDS = OrderedDict(
    (
        ("ELAISS1", SkyCoord("00h37m48s", "-44d00m00s")),
        ("XMM-LSS", SkyCoord("02h22m50s", "-04d45m00s")),
        ("ECDFS", SkyCoord("03h32m30s", "-28d06m00s")),
        ("COSMOS", SkyCoord("10h00m24s", "+02d10m55s")),
        ("EDFS", SkyCoord("03h55m52.8s", "-49d16m48s")),
        ("EDFS2", SkyCoord("04h14m24s", "-47d36m00s")),
    )
)

# exception classes

# interface functions


def main():
    """Make the sample cadence figure."""
    plotprep.config_logging()
    plotprep.config_matplotlib()

    logging.info("Making moon plot")
    fig, ax = plt.subplots()
    plot_hourglass_from_func(sun_alt, YEAR, MONTH, ax=ax)
    logging.info("Saving moon plot")
    fig.savefig(MOON_PLOT_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)

    logging.info("Reading %s", SIM_DATABASE_FNAME)
    with sqlite3.connect(SIM_DATABASE_FNAME) as con:
        visits = pd.read_sql_query("SELECT * FROM SummaryAllProps", con)

    logging.info("Making block plot")
    fig, ax = plt.subplots()
    plot_hourglass_from_blocks(visits, YEAR, MONTH, ax=ax)
    logging.info("Saving block plot")
    fig.savefig(BLOCK_PLOT_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)

    logging.info("GLORIOUS SUCCESS")
    return 0


def plot_hourglass_from_func(
    func,
    year,
    month,
    freq="5min",
    tz="Chile/Continental",
    site=EarthLocation.of_site("Cerro Pachon"),
    max_sun_alt=-8 * u.deg,
    cmap=plt.get_cmap("viridis"),
    ax=None,
):
    """Return visits centered near a pointing

    Parameters
    ----------
    func : `Callable`
        Callable to generate color values.
    freq : `str`
        A pandas frequency string
    year : `int`
        The year for which to make a plot
    month : `int`
        The month number (1-12) for which to make a plot
    tz : `str`
        The observatory timezone
    site : `astropy.coordinates.earth.EarthLocation`
        The observatory site
    max_sun_alt : `astropy.units.quantity.Quantity`
        The sun altitude to define the edges of the night.
    cmap : `matplotlib.colors.Colormap`
        The colormap to use
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axis on which to plot.

    Returns
    -------
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axis on which the hourglass is plotted.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Build the time series of sampled times
    start_date = pd.Timestamp(year=year, month=month, day=1, hour=12, tz=tz)
    end_date = start_date + pd.DateOffset(months=1)

    # Get MJDs
    local_times = pd.date_range(start_date, end_date, freq=freq)[:-1]

    utc_times = local_times.tz_convert("UTC")
    mjds = utc_times.to_julian_date() - 2400000.5

    night_mjds = (
        np.floor(local_times.to_julian_date() - 2400001).astype(int).values
    )
    hours_after_midnight = 24 * (
        local_times.to_julian_date().values - 2400001.5 - night_mjds
    )
    time_width = np.diff(hours_after_midnight)[0]

    if max_sun_alt is not None:
        ap_times = astropy.time.Time(utc_times, location=site)
        night_mask = (
            astropy.coordinates.get_sun(ap_times)
            .transform_to(
                astropy.coordinates.AltAz(obstime=ap_times, location=site)
            )
            .alt
        ) < max_sun_alt
        local_times = local_times[night_mask]
        mjds = mjds[night_mask]
        hours_after_midnight = hours_after_midnight[night_mask]
        night_mjds = night_mjds[night_mask]

    values = func(mjds)
    if values.max() == values.min():
        colors = cmap(0.5)
    else:
        colors = cmap((values - values.min()) / (values.max() - values.min()))

    if fig is not None:
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        fig.colorbar(scalar_mappable)

    ax.bar(
        hours_after_midnight,
        height=0.8,
        width=time_width,
        bottom=night_mjds - np.min(night_mjds) + 0.5,
        color=colors,
        align="edge",
    )
    ax.set_ylim(ax.get_ylim()[-1], -0.5)
    ax.set_title(local_times.month_name()[0])
    ax.set_yticks(np.arange(1, 32, 7))

    return ax


def plot_hourglass_from_blocks(
    visits,
    year,
    month,
    tz="Chile/Continental",
    site=EarthLocation.of_site("Cerro Pachon"),
    max_sun_alt=-8 * u.deg,
    cmap=plt.get_cmap("viridis"),
    legend=True,
    ax=None,
):
    """Return visits centered near a pointing

    Parameters
    ----------
    visits : `pandas.DataFrame`
        Table of visits from opsim.
    year : `int`
        The year for which to make a plot
    month : `int`
        The month number (1-12) for which to make a plot
    tz : `str`
        The observatory timezone
    site : `astropy.coordinates.earth.EarthLocation`
        The observatory site
    max_sun_alt : `astropy.units.quantity.Quantity`
        The sun altitude to define the edges of the night.
    cmap : `matplotlib.colors.Colormap`
        The colormap to use
    legend : `bool`
        Show a legend?
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axis on which to plot.

    Returns
    -------
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The axis on which the hourglass is plotted.
    """
    block_map = {n: note_to_block(n) for n in visits.note.unique()}

    start_date = pd.Timestamp(year=year, month=month, day=1, hour=12, tz=tz)
    start_mjd = start_date.to_julian_date() - 2400000.5
    end_date = start_date + pd.DateOffset(months=1)
    end_mjd = end_date.to_julian_date() - 2400000.5
    month_visits = visits.query(
        f"{start_mjd}<observationStartMJD<{end_mjd}"
    ).copy()
    month_visits["block"] = month_visits.note.map(block_map)
    month_visits["night_mjd"] = np.floor(
        month_visits["observationStartMJD"] + (site.lon.deg / 360) - 0.5
    ).astype(int)
    month_visits["block_num"] = (
        month_visits["block"] != month_visits["block"].shift()
    ).cumsum()
    month_visits["duration"] = (
        month_visits["observationStartMJD"].shift(-1)
        - month_visits["observationStartMJD"]
    )
    month_visits.loc[
        month_visits.duration > (5 * u.min).to_value(u.day), "duration"
    ] = (30 * u.second).to_value(u.day)
    month_blocks = (
        month_visits.groupby(["night_mjd", "block_num", "block"])
        .agg(
            {
                "observationStartMJD": "min",
                "fieldRA": "median",
                "duration": "sum",
                "moonDistance": "median",
            }
        )
        .sort_values("block_num")
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    bar_height = 0.8

    # Get MJDs
    block_datetimes = pd.to_datetime(
        month_blocks.observationStartMJD.values + 2400000.5,
        unit="D",
        origin="julian",
    ).tz_localize("UTC")
    local_times = pd.DatetimeIndex(block_datetimes).tz_convert(
        "Chile/Continental"
    )

    utc_times = local_times.tz_convert("UTC")
    mjds = month_blocks.observationStartMJD.values

    (
        night_mjds,
        month_blocks["hours_after_midnight"],
    ) = compute_hours_after_midnight(mjds)

    block_types = month_blocks.reset_index().block.unique()
    block_cmap = make_block_cmap(block_types, "Set2")
    for block_type in block_types:
        these_blocks = month_blocks.query(f'block=="{block_type}"')
        ax.bar(
            these_blocks["hours_after_midnight"],
            height=bar_height,
            width=(these_blocks["duration"].values * u.day).to_value(u.hour),
            bottom=these_blocks.reset_index()["night_mjd"] - start_mjd + 1,
            color=block_cmap[block_type],
            align="edge",
            label=block_type,
        )

    if legend:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(ax.get_ylim()[-1], -0.5)
    ax.set_title(local_times.month_name()[0])
    ax.set_yticks(np.arange(1, 32, 7))
    xlim = ax.get_xlim()

    for block_type in set(DEEP_COORDS).intersection(block_types):
        ra = DEEP_COORDS[block_type].ra.deg
        transit_mjds = compute_coord_transit_mjds(
            np.arange(np.floor(mjds.min()), np.ceil(mjds.max() + 2)), ra, site
        )
        transit_nights, transit_hams = compute_hours_after_midnight(
            transit_mjds
        )
        ax.plot(
            transit_hams,
            transit_nights - start_mjd + 1,
            color=block_cmap[block_type],
        )

    moon_transit_mjds = compute_moon_transit_mjds(
        np.arange(np.floor(mjds.min()), np.ceil(mjds.max() + 2)), site
    )
    moon_transit_nights, moon_transit_hams = compute_hours_after_midnight(
        moon_transit_mjds
    )
    # Loop to avoid wrapping
    moon_lines = np.cumsum(
        np.diff(moon_transit_hams, prepend=moon_transit_hams[0]) < 0
    )
    for moon_line in np.unique(moon_lines):
        these_hams = moon_transit_hams[moon_lines == moon_line]
        these_nights = moon_transit_nights[moon_lines == moon_line]
        ax.plot(
            these_hams,
            these_nights - start_mjd + 1,
            color="yellow",
            linewidth=8,
            alpha=0.5,
        )

    ax.set_xlim(xlim)


# classes

# internal functions & classes


def sun_alt(mjd, site=EarthLocation.of_site("Cerro Pachon")):
    t = Time(mjd, format="mjd")
    sun_alt = (
        astropy.coordinates.get_sun(t)
        .transform_to(astropy.coordinates.AltAz(obstime=t, location=site))
        .alt.deg
    )
    return sun_alt


def moon_alt(mjd, site=EarthLocation.of_site("Cerro Pachon")):
    t = Time(mjd, format="mjd")
    moon_alt = (
        astropy.coordinates.get_moon(t)
        .transform_to(astropy.coordinates.AltAz(obstime=t, location=site))
        .alt.deg
    )
    return moon_alt


def mean_airmass(mjds, visits):
    airmass = visits.groupby(pd.cut(visits.observationStartMJD, mjds))[
        "airmass"
    ].mean()
    return airmass


def visit_count(mjds, visits):
    counts = visits.groupby(pd.cut(visits.observationStartMJD, mjds))[
        "observationId"
    ].count()
    return counts


def compute_coord_transit_mjds(mjds, ra, site):
    lsts = (
        astropy.time.Time(mjds, format="mjd", location=site)
        .sidereal_time("apparent")
        .deg
    )
    ha = Angle(lsts - ra, unit=u.deg).wrap_at(180 * u.deg)
    mjds = mjds - ha.to_value(u.deg) / 360
    return mjds


def compute_moon_transit_mjds(mjds, site):
    for iter in np.arange(3):
        times = astropy.time.Time(mjds, format="mjd", location=site)
        moon_coords = astropy.coordinates.get_moon(times)
        lsts = astropy.time.Time(
            mjds, format="mjd", location=site
        ).sidereal_time("apparent")
        ha = (lsts - moon_coords.ra).wrap_at(180 * u.deg)
        mjds = mjds - ha.to_value(u.deg) / 360
    return mjds


def note_to_block(note):
    visible_bands = ("u", "g", "r")
    note_elems = note.replace(":", ", ").split(", ")
    if note_elems[0] == "greedy":
        return note_elems[0]
    elif note_elems[0] == "DD":
        return note_elems[1]
    elif note_elems[0] == "blob":
        for band in visible_bands:
            if band in note_elems[1]:
                return "u,g,r"
        return "IR"
    else:
        assert False


def make_block_cmap(block_types, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    cmap_idx = 0
    block_cmap = OrderedDict(
        (("IR", "darkred"), ("u,g,r", "darkblue"), ("greedy", "black"))
    )
    for block_type in block_types:
        if block_type not in block_cmap:
            block_cmap[block_type] = cmap(cmap_idx)
            cmap_idx += 1
    return block_cmap


def compute_hours_after_midnight(mjd, tz="Chile/Continental"):
    utc_datetimes = pd.to_datetime(
        mjd + 2400000.5, unit="D", origin="julian"
    ).tz_localize("UTC")
    local_times = pd.DatetimeIndex(utc_datetimes).tz_convert(tz)
    night_mjds = (
        np.floor(local_times.to_julian_date() - 2400001).astype(int).values
    )
    hours_after_midnight = (
        local_times.to_julian_date().values - 2400001.5 - night_mjds
    ) * 24
    return night_mjds, hours_after_midnight


if __name__ == "__main__":
    sys.exit(main())
