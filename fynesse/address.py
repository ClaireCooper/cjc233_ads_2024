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
from pyproj import Transformer

"""Address a particular question that arises from the data"""


def get_output_area_from_coordinates(conn, longitude, latitude):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700",  always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    query = (f'SELECT output_area FROM oa_data '
             f'WHERE ST_CONTAINS(geometry, ST_GeomFromText("POINT({easting} {northing})")) LIMIT 1')
    df = pd.read_sql(query, conn)
    return df['output_area']
