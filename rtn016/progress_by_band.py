"""Create a plot of progress by band"""

# imports
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from . import plotprep
import plotprep

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db

# constants

SIM_DATABASE_FNAME = "/data/des91.b/data/neilsen/LSST/devel/sim-data/sims_featureScheduler_runs1.7/baseline/baseline_nexp2_v1.7_10yrs.db"
FIGURE_FNAME = "figures/progress_by_band.png"
NUM_NIGHTS = 183

# exception classes

# interface functions


def main():
    """Make the sample cadence figure.
    """
    plotprep.config_logging()
    plotprep.config_matplotlib()

    logging.info("Making plot")
    fig, axes = plot_progress_by_band()
    logging.info("Saving plot")
    fig.savefig(FIGURE_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)
    logging.info("GLORIOUS SUCCESS")
    return 0


def plot_progress_by_band():
    """Plot LSST progress by band.

    Parameters
    ----------

    Returns
    -------
    fig : the `matplotlib.figure.Figure`
        the figure with the plot
    axes : `list(matplotlib.axes.Axes)`
        the axes in the plot
    """
    ops_db = lsst.sims.maf.db.OpsimDatabase(SIM_DATABASE_FNAME)
    vector_bins = np.arange(365 * 10 + 2)
    metric = metrics.AccumulateCountMetric(
        col="observationStartMJD", bins=vector_bins, binCol="night"
    )
    slicer = slicers.UniSlicer()
    bundles = {}
    for band in plotprep.BANDS:
        sql = f"filter = '{band}'"
        bundles[band] = metricBundles.MetricBundle(
            metric, slicer, sql, plotDict={}, plotFuncs=[], summaryMetrics=[]
        )

    sql = ""
    slicer = slicers.OneDSlicer(sliceColName="night", bins=vector_bins)
    metric = metrics.MeanMetric(col="moonPhase")
    bundles["moon"] = metricBundles.MetricBundle(metric, slicer, sql)

    sql = ""
    slicer = slicers.OneDSlicer(sliceColName="night", bins=vector_bins)
    metric = metrics.MeanMetric(col="observationStartMJD")
    bundles["mjd"] = metricBundles.MetricBundle(metric, slicer, sql)
    metric_group = metricBundles.MetricBundleGroup(bundles, ops_db)

    metric_group.runAll()

    fig, ax = plt.subplots()
    num_nights = NUM_NIGHTS

    mjd = bundles["mjd"].metricValues[:num_nights]

    for band in plotprep.BANDS:
        bundle = bundles[band]
        ax.plot(
            mjd,
            bundle.metricValues[0, :num_nights],
            c=plotprep.BAND_COLOR[band],
            label=band,
        )

    ax.scatter(
        mjd,
        np.zeros(num_nights),
        c=bundles["moon"].metricValues[:num_nights],
        cmap="cividis",
        s=5,
    )
    ax.legend()

    start_date = pd.to_datetime(
        mjd.min() - 15 + 2400000.5, unit="D", origin="julian"
    )
    end_date = pd.to_datetime(
        mjd.max() + 15 + 2400000.5, unit="D", origin="julian"
    )
    date_seq = pd.date_range(start=start_date, end=end_date, freq="MS")
    ax.set_xticks(date_seq.to_julian_date() - 2400000.5)
    ax.set_xticklabels(str(d)[:10] for d in date_seq)

    ax.set_ylabel("Number of visits")

    return fig, ax


# classes

# internal functions & classes

if __name__ == "__main__":
    sys.exit(main())
