from .config import *

from . import access

import pandas as pd
import matplotlib.pyplot as plt

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, 
how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. 
Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly 
timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct,
    column names informative, date and times correctly formatted. Return a structured data structure such as a data
    frame."""
    df = access.data()
    raise NotImplementedError


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def buildings_with_addresses(buildings_df):
    return buildings_df[
        (buildings_df['addr:housenumber'].notnull() | buildings_df['addr:housename'].notnull()) & buildings_df[
            'addr:street'].notnull() & buildings_df['addr:postcode'].notnull()].loc[
        'way', ['addr:housenumber', 'addr:housename', 'addr:street', 'addr:postcode', 'geometry']]


def areas_from_gdf(gdf):
    return gdf['geometry'].to_crs(epsg=6933).area


def plot_features_on_axis(axis, gdf):
    gdf.plot(ax=axis, linewidth=1, edgecolor="dimgray")


def plot_gdfs(gdfs):
    plt.rcParams['axes.formatter.useoffset'] = False
    fig, ax = plt.subplots()
    for gdf in gdfs:
        plot_features_on_axis(ax, gdf)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    plt.tight_layout()
    plt.show()
