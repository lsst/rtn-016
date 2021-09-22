"""Create sample boxplots showing depth vs. time."""

# imports
import sys
from collections import OrderedDict, namedtuple

import sqlite3
import numpy as np
import healpy

import matplotlib.pyplot as plt

import lsst.sims.maf
import lsst.sims.maf.db
import lsst.sims.maf.metrics
import lsst.sims.maf.slicers
import lsst.sims.maf.stackers
import lsst.sims.maf.plots
import lsst.sims.maf.metricBundles

try:
    import plotprep
except:
    from . import plotprep
    
# constants

SIM_DATABASE_FNAME = "/data/des91.b/data/neilsen/LSST/devel/sim-data/sims_featureScheduler_runs1.7/baseline/baseline_nexp2_v1.7_10yrs.db"
BAND = "i"
NSIDE = 64
FOOTPRINT_AREA_DEG = 18000

# exception classes

# interface functions


def create_metric_by_year_boxplot(
        this_metric=lsst.sims.maf.metrics.Coaddm5Metric(),
        bin_width = 0.25,
        right_y = None
):
    """Create a boxplot of limiting magnitude by time (years)."""

    mjd_values = OrderedDict()
    mjd_range = _query_mjd_range()
    for year in np.arange(bin_width, 10 + bin_width, bin_width):
        mjd_values[year] = np.round(mjd_range.min + 365.242 * year + 0.2) - 0.2

    depths_at_times = _compute_metric_by_time(
        mjd_values, this_metric=this_metric
    )

    # Only include 18000 sq. deg. that are best in the final footprint.
    in_footprint = _find_footprint_hpix()

    fig, ax = plt.subplots()

    def compute_right_ax(left_ax):
        y1, y2 = left_ax.get_ylim()
        right_ax.set_ylim(y1/30, y2/30)
        right_ax.figure.canvas.draw()

    if right_y is not None:
        right_ax = ax.twinx()
        ax.callbacks.connect("ylim_changed", compute_right_ax)

    ax.boxplot(
        [d[in_footprint] for d in depths_at_times.values()],
        whis=(5, 95),
        showfliers=False,
    )
    ax.set_xlabel("Year")
    ax.set_xticks(np.arange(11)/bin_width)
    ax.set_xticklabels([int(y) for y in ax.get_xticks()*bin_width])
    if right_y is not None:
        # This is a t_eff plot, so add the right axis and reference scale
        right_ax.grid(False)
        right_ax.set_ylabel(right_y)

        for dmag in (0.2, 0.3):
            teff_max = 30*825*10**(-0.8*dmag) 
            ax.plot([0, 10/bin_width], [0, teff_max])
        
    return fig, ax


def main():
    """Create the sample progress plots"""
    plotprep.config()

    figure_params = (
        (
            "figures/coaddm5_boxplot.pdf",
            "coadd m5 limiting magnitude",
            lsst.sims.maf.metrics.Coaddm5Metric(),
            None,
        ),
        (
            "figures/teff_boxplot.pdf",
            r"Accumulated $t_{\mathrm{eff}}$ (seconds)",
            lsst.sims.maf.metrics.TeffMetric(),
            r"Accumulated $t_{\mathrm{eff}}$ (nominal visits)",
        ),
        (
            "figures/numvisits_boxplot.pdf",
            "Accumulated visits",
            lsst.sims.maf.metrics.CountMetric(col="observationStartMJD"),
            None,
        ),
    )

    for fname, ylabel, metric, right_y in figure_params:
        fig, ax = create_metric_by_year_boxplot(this_metric=metric, right_y=right_y)
        ax.set_ylabel(ylabel)
        fig.savefig(fname, dpi=600, bbox_inches="tight", pad_inches=0)

    return 0


# classes

ValueRange = namedtuple("ValueRange", ("min", "max"))

# internal functions & classes


def _query_mjd_range():
    with sqlite3.connect(SIM_DATABASE_FNAME) as con:
        cursor = con.cursor()
        cursor.execute(
            "SELECT MIN(observationStartMjd), MAX(observationStartMJD) FROM summaryAllProps"
        )
        mjd_range = ValueRange(*cursor.fetchall()[0])
    return mjd_range


def _compute_metric_at_mjd(
    mjd=None, this_metric=lsst.sims.maf.metrics.Coaddm5Metric(), band=BAND
):
    opsdb = lsst.sims.maf.db.OpsimDatabase(SIM_DATABASE_FNAME)
    hpix_slicer = lsst.sims.maf.slicers.HealpixSlicer(
        nside=NSIDE, latLonDeg=True
    )

    if mjd is None:
        query = f'filter="{band}"'
    else:
        query = f'filter="{band}" and observationStartMjd<{mjd}'

    if mjd is None:
        query = ""
    else:
        query = f'observationStartMjd<{mjd}'

        
    metric_bundles = {
        "this_metric": lsst.sims.maf.metricBundles.MetricBundle(
            this_metric, hpix_slicer, query
        )
    }
    mbgroup = lsst.sims.maf.metricBundles.MetricBundleGroup(
        metric_bundles, opsdb
    )
    mbgroup.runAll()
    metric_values = metric_bundles["this_metric"].metricValues
    return metric_values


def _find_footprint_hpix():
    final_metric_values = _compute_metric_at_mjd().filled(0.0)
    num_hpix_in_footprint = int(
        np.ceil(FOOTPRINT_AREA_DEG / healpy.nside2pixarea(NSIDE, degrees=True))
    )

    # multiply by -1 because argpartition gives lowest values, we want highest
    in_footprint = np.argpartition(
        -1 * final_metric_values, num_hpix_in_footprint
    )[:num_hpix_in_footprint]

    if __debug__:
        num_out_of_footprint = len(final_metric_values) - num_hpix_in_footprint
        out_of_footprint = np.argpartition(
            final_metric_values, num_out_of_footprint
        )[:num_out_of_footprint]
        assert np.min(final_metric_values[in_footprint]) >= np.max(
            final_metric_values[out_of_footprint]
        )

    return in_footprint


def _compute_metric_by_time(
    bins,
    this_metric=lsst.sims.maf.metrics.Coaddm5Metric(),
    compute_metric_at_mjd=_compute_metric_at_mjd,
):
    metric_value_by_time = OrderedDict()
    for key, mjd in bins.items():
        metric_value_by_time[key] = compute_metric_at_mjd(
            mjd, this_metric=this_metric
        )

    return metric_value_by_time


if __name__ == "__main__":
    sys.exit(main())
