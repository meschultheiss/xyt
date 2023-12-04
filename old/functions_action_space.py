from functions_action_space import *

# Standard Libraries
import math
from datetime import datetime
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

# Data Manipulation and Analysis
import pandas as pd
import geopandas as gpd
from geopandas import sjoin
import numpy as np
import time
import utm

# Visualization
import plotly.graph_objects as go
import seaborn as sns

sns.set_theme(style="white")
sns.set_palette("Pastel2_r")
cm = sns.light_palette("#fff4ac", as_cmap=True)
from matplotlib import pyplot as plt
import folium

# Point Pattern Analysis and Spatial Statistics
import pointpats
from pointpats.centrography import (
    hull,
    mbr,
    mean_center,
    weighted_mean_center,
    manhattan_median,
    std_distance,
    euclidean_median,
    ellipse,
    dtot,
)

# from pointpats.distance_statistics import euclidean

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from haversine import haversine, Unit
from matplotlib.patches import Ellipse
from pylab import figure, show, rand

from pointpats import PointPattern
from scipy.spatial import distance
from pysal.explore.pointpats import (
    mean_center,
    weighted_mean_center,
    std_distance,
    euclidean_median,
    ellipse,
)
from sklearn.metrics.pairwise import haversine_distances
from geopy.distance import geodesic

import libpysal as ps
from pointpats import PointPattern


# Other settings
pd.set_option("display.max_columns", 999)
from IPython.display import display, HTML

css = """
.output {
    flex-direction: row;
}
"""

HTML("<style>{}</style>".format(css))


def modified_std_distance(pp, center):
    """
    Calculate standard distance of a point array like std_distance() does in PySAL with the mean center, but we can specify here a different center
    pp: point pattern
    center: array([ x, y])
    """
    return np.sqrt(
        1
        / len(pp)
        * (
            np.sum((pp.geometry.x - center[0]) ** 2)
            + np.sum((pp.geometry.y - center[1]) ** 2)
        )
    )


def principal_comp_inspector(
    X, method="famd", n_pc=4, cm=sns.diverging_palette(360, 65, l=80, as_cmap=True)
):
    # cm = sns.light_palette("#fff4ac",as_cmap=True)

    method_ = 0

    if method == "pca":
        # PCA
        pca = prince.PCA(
            n_components=n_pc,
            n_iter=5,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=True,
            check_input=True,
            engine="auto",
            random_state=42,
        )

        PCA_X = pca.fit_transform(X)
        PCA_X.reset_index(inplace=True, drop=True)
        PCA_X.columns = ["PC" + str(pc) for pc in range(1, n_pc + 1)]

        PC_X = PCA_X

        method_ = pca.fit(X)

        loadings = pca.column_correlations(X)
        loadings.columns = ["PC" + str(pc) for pc in range(1, n_pc + 1)]
        loadings.style.background_gradient(cmap=cm)

    elif method == "famd":
        # FAMD
        famd = prince.FAMD(
            n_components=n_pc,
            n_iter=10,
            copy=True,
            check_input=True,
            engine="auto",
            random_state=42,
        )

        FAMD_X = famd.fit_transform(X)
        FAMD_X.reset_index(inplace=True, drop=True)
        FAMD_X.columns = ["PC" + str(pc) for pc in range(1, n_pc + 1)]

        PC_X = FAMD_X

        method_ = famd.fit(X)

        loadings = famd.column_correlations(X)
        loadings.columns = ["PC" + str(pc) for pc in range(1, n_pc + 1)]
        loadings.style.background_gradient(cmap=cm)

    else:
        raise print(
            "Methods implemented are pca or famd, please use a correct method for your dimension reduction"
        )

    df = pd.DataFrame(
        method_.eigenvalues_,
        columns=["eigenvalue"],
        index=["PC" + str(pc) for pc in range(1, n_pc + 1)],
    )
    df["explained_inertia"] = method_.explained_inertia_
    df["cumulative_inertia"] = df.explained_inertia.cumsum()

    return df, PC_X, loadings.style.background_gradient(cmap=cm), method_


def plot_2d_clusters(
    fa, X_, PC1=0, PC2=1, title="Two-dimensional components projection"
):
    fa.plot_row_coordinates(
        X_,
        ax=None,
        figsize=(8, 8),
        x_component=PC1,
        y_component=PC2,
        labels=None,
        # color_labels=['Profile {}'.format(p) for p in labels],
        ellipse_outline=False,
        ellipse_fill=True,
        show_points=True,
    )
    plt.title(title, fontdict={"fontweight": "bold"})

    return plt.show()


def cov_corr_heatmap(
    X,
    method="cov",
    title="",
    annot=False,
    cmap=sns.diverging_palette(360, 65, l=80, as_cmap=True),
):
    """
    Desription:
        Map the correlation, covariance or p-values of a set of observed variables.

    Args:
        X: variables to study
        method: 'corr' for correlation matrix
                'cov' for covariance matrix
                'pval' for p-values
        title: add a personalized [str]
        annot: if True return the values in the heatmap cells

    Returns:
        - Heatmap of correlation, covariance or p-values of X
    """

    if method == "cov":
        X_ = X.cov()
        vmin = None
        title2 = "Covariance matrix of X_"
        # check if data is already normalized
        X2 = pd.DataFrame(
            StandardScaler().fit_transform(
                X.select_dtypes(exclude=["bool", "string", "object"])
            )
        )
        check_norm = (
            round(X.select_dtypes(exclude=["bool", "string", "object"]), 5).to_numpy()
            == round(X2, 5).to_numpy()
        ).all()
        if check_norm:
            print(
                "NB - you are computing the covariance of a normalized set of data. In that case, covariance is equivalent to correlation"
            )
    elif method == "corr":
        X_ = X.corr()
        vmin = None
        title2 = "Correlation matrix of X_"
    elif method == "pval":
        X_ = calculate_pvalues(X)
        vmin = 0.05
        cmap.set_under(color="grey", alpha=0.1)
        title2 = (
            "P-values for matrix X_"
            + "\ngrey cells : p-value < 0.05 i.e. reject H0 with confidence"
            + "\nred cells : p-value > 0.05 i.e. cannot rejet H0 with confidence"
        )
    else:
        raise print("Please implement cov or corr method.")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(X_, dtype=bool), 0)

    # Set the title
    if len(title) < 1:
        title = title2  # pre-set titles else specified titles
    else:
        if method == "pval":
            title = (
                title
                + "\ngrey cells : p-value < 0.05 i.e. reject H0 with confidence"
                + "\nred cells : p-value > 0.05 i.e. cannot rejet H0 with confidence"
            )

    # Set up the matplotlib figure
    f, ax = plt.subplots(ncols=1, figsize=(7, 7))
    plt.title(title)

    # Draw the heatmap with the mask and correct aspect ratio
    fig = sns.heatmap(
        X_,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=None,
        center=0,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.7},
        annot=annot,
        fmt=".2g",
    )
    return plt.show()


def sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    return f


def theta(X, y):
    Xp = np.hstack([np.ones((len(X), 1)), X])  # adds bias column
    theta_ = np.linalg.solve(np.dot(Xp.T, Xp), np.dot(Xp.T, y))
    # theta_ = np.dot(np.linalg.inv(np.dot(Xp.T,Xp)),np.dot(Xp.T,y)) ## Alternative, but more confusing
    # raise NotImplementedError()

    return theta_


def J(X, y, theta):
    Xp = np.hstack([np.ones((len(X), 1)), X])
    m = 1 / (2 * len(X))
    J_ = np.multiply(m, np.sum(np.square(np.dot(Xp, theta) - y)))
    # raise NotImplementedError()

    return J_


def f(N, mu=1.0, sigma=0.5):
    y = np.exp(-((np.log(N) - mu) ** 2) / (2 * sigma**2)) / (
        N * sigma * (2 * np.pi) ** 0.5
    )
    return y
