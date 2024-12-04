# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import colors
from pyproj import Transformer

from . import assess

"""Address a particular question that arises from the data"""


def get_output_area_from_coordinates(conn, longitude, latitude):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    query = (f'SELECT output_area FROM oa_data '
             f'WHERE ST_CONTAINS(geometry, ST_GeomFromText("POINT({easting} {northing})")) LIMIT 1')
    df = pd.read_sql(query, conn)
    return df['output_area'].tolist()[0]


def get_feature_counts_for_output_area(conn, output_area, year, distance=1000):
    db_query = (f'SELECT tagkey, tagvalue, count FROM osm_oa_radius_counts '
                f'WHERE year={year} AND output_area="{output_area}"'
                f'AND distance={distance}')
    df = pd.read_sql(db_query, conn)
    return df


def train_and_save_glm(family_with_link, x, y, path):
    y_glm = sm.GLM(y, x, family=family_with_link)
    y_model = y_glm.fit()
    y_model.save(path)
    print('Saved trained model to:', path)


def plot_area_variable_map(ax, areas, values):
    areas_withy = areas.copy()
    areas_withy = areas_withy.reset_index()
    areas_withy['y'] = values
    areas_withy.plot(ax=ax, column='y', legend=True, edgecolor="face", linewidth=0.2, cmap='viridis_r')
    ax.set_axis_off()


def plot_area_variable_map_log_colorscale(ax, areas, values):
    areas_withy = areas.copy()
    areas_withy = areas_withy.reset_index()
    areas_withy['y'] = values
    areas_withy.plot(ax=ax, column='y', legend=True, edgecolor="face", linewidth=0.3,
                     norm=colors.LogNorm(vmin=areas_withy[areas_withy.y >= 1].y.min(),
                                         vmax=areas_withy.y.max(), clip=True))
    ax.set_axis_off()


def plot_residuals_map(ax, areas, residuals):
    areas_withy = areas.copy()
    areas_withy['residual'] = residuals
    areas_withy.plot(ax=ax, column='residual', legend=True, edgecolor="face", linewidth=0.2, cmap='PRGn')
    ax.set_axis_off()


def get_students(conn, students_df, latitude, longitude):
    output_area = get_output_area_from_coordinates(conn, longitude, latitude)
    return students_df.loc[output_area]['L15']
