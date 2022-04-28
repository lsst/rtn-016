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
FIGURE_FNAME = "figures/progress_by_quadrant.png"
NUM_NIGHTS = 365 * 2

# exception classes

# interface functions


def main():
    """Make the sample cadence figure.
    """
    plotprep.config_logging()
    plotprep.config_matplotlib()

    logging.info("Making plot")
    fig, axes = plot_progress_by_quadrant()
    logging.info("Saving plot")
    fig.savefig(FIGURE_FNAME, dpi=600, bbox_inches="tight", pad_inches=0)
    logging.info("GLORIOUS SUCCESS")
    return 0


def plot_progress_by_quadrant():
    """Plot LSST progress by quadrant of the sky.

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

    vector_bins = np.arange(366 * 10)
    metric = metrics.AccumulateCountMetric(
        col="observationStartMJD", bins=vector_bins, binCol="night"
    )
    slicer = slicers.UniSlicer()
    bundles = {}
    quads = np.arange(0, 360, 90)
    for quad in quads:
        sql = f"fieldRA >= {quad} and fieldRA < {quad+90}"
        bundles[quad] = metricBundles.MetricBundle(
            metric, slicer, sql, plotDict={}, plotFuncs=[], summaryMetrics=[]
        )

    sql = ""
    slicer = slicers.OneDSlicer(sliceColName="night", bins=vector_bins)
    metric = metrics.MeanMetric(col="observationStartMJD")
    bundles["mjd"] = metricBundles.MetricBundle(metric, slicer, sql)
    metric_group = metricBundles.MetricBundleGroup(bundles, ops_db)
    metric_group.runAll()

    fig, ax = plt.subplots()
    num_nights = NUM_NIGHTS
    mjd = bundles["mjd"].metricValues[:num_nights]
    for quad in quads:
        bundle = bundles[quad]
        ax.plot(
            mjd,
            bundle.metricValues[0, :num_nights],
            label=f"{quad}$^\circ$ $\leq$ R.A. < {quad+90}$^\circ$",
        )

    ax.legend()
    start_date = pd.to_datetime(
        mjd.min() - 15 + 2400000.5, unit="D", origin="julian"
    )
    end_date = pd.to_datetime(
        mjd.max() + 15 + 2400000.5, unit="D", origin="julian"
    )
    date_seq = pd.date_range(start=start_date, end=end_date, freq="Q")
    ax.set_xticks(date_seq.to_julian_date() - 2400000.5)
    ax.set_xticklabels([str(d)[:10] for d in date_seq], rotation=15)

    ax.set_ylabel("Number of visits")

    return fig, ax


# classes

# internal functions & classes

if __name__ == "__main__":
    sys.exit(main())
