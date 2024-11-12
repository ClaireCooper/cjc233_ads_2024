from .config import *

from . import access

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math

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


def join_ppaid_and_osm_address_dfs(ppaid_df, address_area_gdf):
    address_area_gdf['upper_street'] = address_area_gdf['addr:street'].str.upper().str.replace("'", "")
    joined_df = ppaid_df.merge(address_area_gdf, left_on='street', right_on='upper_street')
    joined_df = joined_df[
        (joined_df['primary_addressable_object_name'].str.upper() == joined_df['addr:housenumber'].str.upper()) | (
                joined_df['primary_addressable_object_name'].str.upper() == joined_df['addr:housename'].str.upper())
        ]
    return joined_df.drop('upper_street', axis=1)


def scatter_plot(ax, x, y, title, x_label, y_label, x_log, y_log):
    ax.plot(x, y, 'bx')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if x_log:
        ax.set_xscale('log')
    ax.set_ylabel(y_label)
    if y_log:
        ax.set_yscale('log')


def plot_factors_affecting_price(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    scatter_plot(ax1,
                 df['area_m2'],
                 df['price'],
                 'Area vs price', 'area (log)',
                 'price(log)',
                 True, True)

    ax2.boxplot((
        df[df['tenure_type'] == 'L']['price'],
        df[df['tenure_type'] == 'F']['price']),
        showmeans=False, showfliers=False,
        labels=['Leasehold', 'Freehold'])
    ax2.set_title('Lease type vs price')
    ax2.set_ylabel('price')

    scatter_plot(ax3,
                 df[df['tenure_type'] == 'F']['date_of_transfer'],
                 df[df['tenure_type'] == 'F']['price'],
                 'Date of transfer vs price for freeholds only',
                 'date',
                 'price(log)',
                 False, True)

    scatter_plot(ax4,
                 df[df['tenure_type'] == 'F']['area_m2'],
                 df[df['tenure_type'] == 'F']['price'],
                 'Area vs price for freeholds only',
                 'area (log)',
                 'price(log)',
                 True, True)

    plt.show()


def predict_prices_now(timestamps, prices):
    log_prices = list(map(lambda p: math.log(p), prices.tolist()))
    price_model = LinearRegression().fit(timestamps, log_prices)
    m = price_model.coef_
    c = price_model.intercept_
    # ln(p) = mt + c
    # p = e^(mt+c)

    log_predicted_price_today = (
        price_model.predict([[datetime.combine(date.today(), datetime.min.time()).timestamp()]])[0])

    return list(map(lambda tp: math.exp(tp[1] + (log_predicted_price_today - price_model.predict([tp[0]])[0])),
                    zip(timestamps, log_prices)))


def timestamp_from_date(date):
    return datetime.combine(date, datetime.min.time()).timestamp()


def add_predict_prices_to_price_paid_df(df):
    timestamps = df['date_of_transfer'].apply(lambda d: [timestamp_from_date(d)])

    df['predicted_price_now'] = predict_prices_now(timestamps.tolist(), df['price'])


def plot_area_adjusted_correlations(all_data_df, freeholds_df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 5))
    scatter_plot(ax1,
                 all_data_df['area_m2'],
                 all_data_df['price'],
                 'Area vs price', 'area (log)',
                 'price(log)',
                 True, True)

    scatter_plot(ax2,
                 freeholds_df['area_m2'],
                 freeholds_df['price'],
                 'Area vs price for freeholds only',
                 'area (log)',
                 'price(log)',
                 True, True)

    scatter_plot(ax3,
                 freeholds_df['area_m2'],
                 freeholds_df['predicted_price_now'],
                 'Area vs estimated 2024 price\n for freeholds only',
                 'area (log)',
                 '2024 price(log)',
                 True, True)

    plt.show()
