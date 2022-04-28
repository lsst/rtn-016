"""Map the airmass"""

# imports
import sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import palpy
import healpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy
import astropy.time
import astropy.coordinates
import astropy.units as u
import astropy.timeseries

# constants
FIGURE_FNAME = "figures/airmass_map.png"
PLANCK_FNAME = "/data/des70.a/data/neilsen/obstac_note/data/HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
YBS_FNAME = "/data/des70.a/data/neilsen/obstac_note/data/bsc5.dat"
EBMV_COL_IDX = 2
SITE = astropy.coordinates.EarthLocation.of_site("Cerro Pachon")

# exception classes

# interface functions


def load_bright_stars(fname):
    ybs_columns = OrderedDict(
        (
            ("HR", (0, 4)),
            ("name", (4, 14)),
            ("RA_hour", (75, 77)),
            ("RA_min", (77, 79)),
            ("RA_sec", (79, 83)),
            ("decl_sign", (83, 84)),
            ("decl_deg", (84, 86)),
            ("decl_min", (86, 88)),
            ("decl_sec", (88, 90)),
            ("Vmag", (102, 107)),
        )
    )
    bs = pd.read_fwf(
        fname,
        colspecs=[ybs_columns[k] for k in ybs_columns],
        names=[k for k in ybs_columns],
    )
    bs["ra"] = (360 / 24) * (
        bs.RA_hour + (bs.RA_min + bs.RA_sec / 60.0) / 60.0
    )
    bs["decl"] = bs.decl_deg + (bs.decl_min + bs.decl_sec / 60.0) / 60.0
    southern_stars = bs.decl_sign == "-"
    bs.loc[southern_stars, "decl"] = -1 * bs.loc[southern_stars, "decl"]
    return bs


def load_dust(fname):
    gal_dust = healpy.read_map(PLANCK_FNAME, EBMV_COL_IDX)
    gal_ecl_rotator = healpy.Rotator(coord=["G", "E"])
    ecl_dust = gal_ecl_rotator.rotate_map_pixel(gal_dust)
    return ecl_dust


def compute_airmass_limit_coords(obs_time, airmass):
    decl = np.arange(-90, 90, 0.01)
    lst_deg = obs_time.sidereal_time("apparent").deg

    # Invert Kristensen's formula for airmass (https://doi.org/10.1002/asna.2123190313)
    a = 470.0
    mu = (1.0 / airmass) + (1 / (2.0 * a)) * ((1.0 / airmass) - airmass)
    cosha = mu / (
        np.cos(np.radians(SITE.lat.deg)) * np.cos(np.radians(decl))
    ) - np.tan(np.radians(SITE.lat.deg)) * np.tan(np.radians(decl))
    ha_deg = np.degrees(np.arccos(cosha))

    rise_ra = lst_deg - ha_deg
    set_ra = lst_deg + ha_deg
    airmass_limit_coords = pd.DataFrame(
        {
            "decl": np.concatenate([decl, np.flip(decl)]),
            "ra": np.concatenate([rise_ra, np.flip(set_ra)]),
        }
    ).dropna()
    return airmass_limit_coords


def compute_twilight_limit_coords(obs_time, zd):
    decl = np.arange(-90, 90, 0.01)
    lst_deg = obs_time.sidereal_time("apparent").deg

    mu = np.cos(np.radians(zd))
    cosha = mu / (
        np.cos(np.radians(SITE.lat.deg)) * np.cos(np.radians(decl))
    ) - np.tan(np.radians(SITE.lat.deg)) * np.tan(np.radians(decl))
    ha_deg = np.degrees(np.arccos(cosha))

    rise_ra = lst_deg - ha_deg
    set_ra = lst_deg + ha_deg
    twilight_coords = pd.DataFrame(
        {
            "decl": np.concatenate([decl, np.flip(decl)]),
            "ra": np.concatenate([rise_ra, np.flip(set_ra)]),
        }
    ).dropna()
    return twilight_coords


def compute_ecliptic(obs_time):
    times = astropy.timeseries.TimeSeries(
        time_start=obs_time, time_delta=1 * u.day, n_samples=365
    )
    coords = astropy.coordinates.get_sun(times.time)
    return coords


def main():
    obs_time = astropy.time.Time("2024-04-01T02:00:00Z", location=SITE)
    airmass_limit = 1.5
    stars = load_bright_stars(YBS_FNAME)
    dust = load_dust(PLANCK_FNAME)

    fig = make_airmass_map(obs_time, airmass_limit, dust, stars)
    fig.savefig(FIGURE_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)


def make_airmass_map(obs_time, airmass_limit, dust, stars):
    solar_coordinates = astropy.coordinates.get_sun(obs_time)
    lunar_coordinates = astropy.coordinates.get_moon(obs_time)
    airmass_limit_coords = compute_airmass_limit_coords(
        obs_time, airmass_limit
    )
    twilight_coords = compute_twilight_limit_coords(obs_time, 90 + 12)
    ecliptic_coords = compute_ecliptic(obs_time)

    truncated_blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "truncated_blues", plt.get_cmap("Blues")(np.linspace(0, 0.3, 100))
    )

    healpy.azeqview(
        dust,
        norm="log",
        min=0.1,
        max=0.3,
        lamb=True,
        rot=(0, -90, 0),
        cbar=False,
        flip="astro",
        reso=18,
        cmap=truncated_blue_cmap,
        title=f"{str(obs_time)[:16]}",
    )
    for vmag in np.arange(4):
        mag_stars = stars.query(f'Vmag<{vmag} and decl<40')
        healpy.projplot(
            mag_stars.ra,
            mag_stars.decl,
            coord="E",
            lonlat=True,
            linestyle="",
            marker="*",
            color="black",
            alpha=0.2*vmag/4.0,
            markersize=(4-vmag),
        )
    healpy.projplot(
        solar_coordinates.ra.deg,
        solar_coordinates.dec.deg,
        coord="E",
        lonlat=True,
        linestyle="",
        marker="*",
        color="green",
        markersize=10,
    )
    healpy.projplot(
        ecliptic_coords.ra.deg,
        ecliptic_coords.dec.deg,
        coord="E",
        lonlat=True,
        alpha=0.5,
        color="green",
    )
    healpy.projplot(
        lunar_coordinates.ra.deg,
        lunar_coordinates.dec.deg,
        coord="E",
        lonlat=True,
        linestyle="",
        marker="o",
        color="orange",
        markersize=10,
    )
    healpy.projplot(
        airmass_limit_coords.ra,
        airmass_limit_coords.decl,
        coord="E",
        lonlat=True,
        color="darkblue",
    )
    healpy.projplot(
        twilight_coords.ra,
        twilight_coords.decl,
        coord="E",
        lonlat=True,
        color="orange",
    )
    healpy.graticule(coord="E")

    fig = plt.gcf()
    return fig


# classes

# internal functions & classes

if __name__ == "__main__":
    status = main()
    sys.exit(status)
