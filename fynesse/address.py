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
import pandas as pd
import statsmodels.api as sm
from pyproj import Transformer

from . import assess

"""Address a particular question that arises from the data"""


def y_pred_ols(x, y, x_pred, design_fn, alpha=0.05, silent=False):
    y_glm = sm.OLS(y, design_fn(x))
    y_model = y_glm.fit()
    if not silent:
        print(y_model.summary())
    return y_model.get_prediction(design_fn(x_pred)).summary_frame(alpha=alpha)


def y_pred(family_with_link, x, y, x_pred, design_fn, alpha=0.05, silent=False):
    y_glm = sm.GLM(y, design_fn(x), family=family_with_link)
    y_model = y_glm.fit()
    if not silent:
        print(y_model.summary())
    return y_model.get_prediction(design_fn(x_pred)).summary_frame(alpha=alpha)


def get_output_area_from_coordinates(conn, longitude, latitude):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700",  always_xy=True)
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
