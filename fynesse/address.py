# This file contains code for supporting addressing questions in the data

import pandas as pd
import statsmodels.api as sm
from pyproj import Transformer

from . import access

"""Address a particular question that arises from the data"""


def get_output_area_from_coordinates(conn, longitude, latitude):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    query = (f'SELECT output_area FROM oa_data '
             f'WHERE ST_CONTAINS(geometry, ST_GeomFromText("POINT({easting} {northing})")) LIMIT 1')
    df = pd.read_sql(query, conn)
    return df['output_area'].tolist()[0]


def train_and_save_glm(family_with_link, x, y, path):
    y_glm = sm.GLM(y, x, family=family_with_link)
    y_model = y_glm.fit()
    y_model.save(path)
    print('Saved trained model to:', path)


def get_students(conn, latitude, longitude):
    students_df = access.select_all_from_table(conn, 'sec_data')
    output_area = get_output_area_from_coordinates(conn, longitude, latitude)
    return students_df.set_index('geography').loc[output_area]['L15']


def get_pd(conn, latitude, longitude):
    pd_df = access.select_all_from_table(conn, 'pd_data')
    output_area = get_output_area_from_coordinates(conn, longitude, latitude)
    return pd_df.set_index('geography').loc[output_area]['population_density']


def get_health(conn, latitude, longitude):
    health_df = access.select_all_from_table(conn, 'health_data')
    output_area = get_output_area_from_coordinates(conn, longitude, latitude)
    return health_df.set_index('geography').loc[output_area]['average']
