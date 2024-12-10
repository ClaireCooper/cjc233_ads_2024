import math
from datetime import datetime, date

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import osmnx.utils_geo
import pandas as pd
import statsmodels.api as sm
from kneed import KneeLocator
from matplotlib import colors
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from . import access

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


def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    bbox = osmnx.utils_geo.bbox_from_point((latitude, longitude), int(distance_km * 1000))
    pois = ox.features_from_bbox(bbox=bbox, tags=tags)
    pois_df = pd.DataFrame(pois)
    poi_counts = {}
    for tag, values in tags.items():
        if tag in pois_df.columns:
            if values is True:
                poi_counts[tag] = pois_df[tag].notnull().sum()
            else:
                for v in values:
                    poi_counts[tag + ':' + v] = len(pois_df[pois_df[tag] == v])
        elif values is True:
            poi_counts[tag] = 0
        else:
            for v in values:
                poi_counts[tag + ':' + v] = 0
    return poi_counts


def normalise_df(df):
    norm = df.copy()
    return (norm - norm.mean()) / norm.std()


def ideal_num_clusters_for_normalised_df(df, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(df.values)
        inertias.append(kmeans.inertia_)

    elbow = KneeLocator(range(1, max_clusters + 1), inertias, curve="convex", direction="decreasing").elbow
    plt.plot(range(1, max_clusters + 1), inertias)
    plt.xticks(range(1, max_clusters + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    return elbow


def distance_matrix_from_normalised_df(df, distance_fun):
    distance_matrix = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
    for x in df.index:
        for y in df.index:
            distance_matrix.loc[x, y] = distance_fun(x, y)
    return distance_matrix


def plot_distance_matrix(fig, ax, matrix):
    im = ax.matshow(matrix)
    axis = np.arange(len(matrix.index))
    ax.set_xticks(axis)
    ax.set_yticks(axis)
    ax.set_xticklabels(matrix.index, fontsize=14, rotation=90)
    ax.set_yticklabels(matrix.index, fontsize=14)
    fig.colorbar(im, ax=ax)


def plot_lat_lon_points_by_category(ax, category_column, gdf):
    gdf.plot(column=category_column, ax=ax, markersize=10, legend=True, categorical=True)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def plot_country_border(ax, gdf):
    gdf.plot(ax=ax, color='white', edgecolor='black')


def gdf_from_df_with_lat_lon(df, lat_column='Latitude', lon_column='Longitude'):
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_column], df[lat_column]))


def select_output_areas_in_limits(conn, north, south, east, west, table_name='oa_data', geometry_column='geometry'):
    db_query = (f'SELECT *, ST_AsBinary({geometry_column}) as geometry_bin FROM {table_name} '
                f'WHERE latitude BETWEEN {south} AND {north} AND longitude BETWEEN {west} AND {east}')
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry_bin'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:4326')
    return gdf.loc[:, ~df.columns.duplicated()].drop('geometry_bin', axis=1)


def select_osm_by_tag_and_value(conn, key, value):
    db_query = (f'SELECT *, ST_AsBinary(geometry) as geometry_bin '
                f'FROM osm_data WHERE tagkey="{key}" AND tagvalue="{value}"')
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry_bin'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:4326')
    return gdf.loc[:, ~df.columns.duplicated()].drop('geometry_bin', axis=1)


def select_osm_by_tag(conn, key):
    db_query = f'SELECT *, ST_AsBinary(geometry) as geometry_bin FROM osm_data WHERE tagkey="{key}"'
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry_bin'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:4326')
    return gdf.loc[:, ~df.columns.duplicated()].drop('geometry_bin', axis=1)


def plot_osm_feature(conn, ax, border, tag, value=None):
    if value is None:
        osm_df = select_osm_by_tag(conn, tag)
        title = tag + ':'
    else:
        osm_df = select_osm_by_tag_and_value(conn, tag, value)
        title = tag + ':' + value
    osm_df['centroid'] = gpd.points_from_xy(osm_df['longitude'], osm_df['latitude'])
    osm_df = osm_df.set_geometry('centroid')
    osm_df = osm_df.set_crs('EPSG:4326')
    osm_df.to_crs(border.crs)
    plot_country_border(ax, border)
    osm_df.plot(ax=ax, markersize=4, color='red')
    ax.set_title(title)
    ax.set_xlim([-6.5, None])
    ax.set_ylim([None, 55.9])
    ax.set_axis_off()


def get_oa_feature_counts(conn, year, features, distance=1000):
    df = pd.read_sql('SELECT output_area FROM census_oa_data ORDER BY output_area', conn)
    for (key, value) in features:
        db_query = (f'SELECT counts.count FROM (SELECT output_area, count FROM osm_oa_radius_counts '
                    f'WHERE year={year} AND tagkey="{key}" AND tagvalue="{value}" and distance={distance}) as counts '
                    f'RIGHT JOIN census_oa_data as oa on oa.output_area = counts.output_area ORDER BY oa.output_area')
        df[key + ':' + value] = pd.read_sql(db_query, conn)['count'].fillna(0).astype(int)
    return df.set_index('output_area')


def get_census_variable(conn, variable):
    db_query = f'SELECT output_area, {variable} FROM census_oa_data ORDER BY output_area'
    df = pd.read_sql(db_query, conn)
    return df.loc[:, ~df.columns.duplicated()].set_index('output_area')


def select_output_area_geometries(conn):
    db_query = f'SELECT output_area, ST_AsBinary(geometry) as geometry FROM oa_data'
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:27700')
    return gdf.loc[:, ~df.columns.duplicated()].set_index('output_area')


def select_random_output_areas(conn, number, seed):
    db_query = (f'SELECT *, ST_AsBinary(geometry) as geo_bin '
                f'FROM oa_data ORDER BY RAND({seed}) LIMIT {number}')
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geo_bin'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:27700')
    return gdf.loc[:, ~df.columns.duplicated()].drop('geo_bin', axis=1).set_index('output_area').sort_index()


def osm_in_oa_radius_counts(conn, output_area, tag, value, distance=1000, year=2021):
    db_query = (f'select count(*) from ('
                f'select latitude, longitude from oa_data where year = {year} '
                f'and output_area="{output_area}") as oa cross '
                f'join (select latitude, longitude, tagkey, tagvalue from osm_data where tagkey="{tag}" and '
                f'tagvalue="{value}") as osm where '
                f'ABS(oa.longitude - osm.longitude) < 0.04 and ABS(osm.latitude - oa.latitude) < 0.04 '
                f'and ST_DISTANCE_SPHERE(POINT(oa.longitude, oa.latitude), '
                f'POINT(osm.longitude, osm.latitude)) < {distance};')
    df = pd.read_sql(db_query, conn)
    return df.values[0][0]


def select_feature_count_for_output_area(conn, output_area, feature, distance=1000, year=2021):
    db_query = (f'SELECT count FROM osm_oa_radius_counts '
                f'WHERE year={year} AND output_area="{output_area}" AND '
                f'tagkey="{feature[0]}" AND tagvalue="{feature[1]}" '
                f'AND distance={distance}')
    with conn.cursor() as cur:
        cur.execute(db_query)
        rows = cur.fetchall()
    if len(rows) == 0:
        return None
    return rows[0][0]


def select_feature_counts(conn, oas_str, feature, year=2021, distance=1000):
    db_query = (f'SELECT count FROM osm_oa_radius_counts '
                f'WHERE year={year} AND output_area IN {oas_str} AND '
                f'tagkey="{feature[0]}" AND tagvalue="{feature[1]}" '
                f'AND distance={distance} ORDER BY output_area')
    with conn.cursor() as cur:
        cur.execute(db_query)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def get_feature_counts(conn, oas, features, year=2021, distance=1000):
    cs = []
    oas = sorted(oas)
    oas_str = '(' + ','.join(['"' + oa + '"' for oa in oas]) + ')'
    for (k, v) in features:
        fcs = select_feature_counts(conn, oas_str, (k, v), year, distance)
        if len(fcs) < len(oas):
            fcs = []
            for oa in oas:
                count = select_feature_count_for_output_area(conn, oa, (k, v), distance, year)
                if count is None:
                    count = osm_in_oa_radius_counts(conn, oa, k, v, distance, year)
                    access.insert_feature_count_for_output_area(conn, oa, k, v, count, distance, year)
                fcs.append(count)
        cs.append(fcs)
    return pd.DataFrame(np.array(cs).T, index=oas, columns=pd.MultiIndex.from_tuples(features))


def get_r2s_for_features(training_features, testing_features, y,
                         family_with_link=sm.families.Gaussian(), design_fns=None, group_size=1):
    r2s = {}
    models = {}
    y_train = y.loc[training_features.index.to_list()]
    y_test = y.loc[testing_features.index.to_list()]
    if design_fns is None:
        design_fns = {'linear': lambda x: x}
    for col in training_features.columns:
        r2s[col] = {}
        models[col] = {}
        x_train = training_features[col]
        x_test = testing_features[col]

        for (name, f) in design_fns.items():
            y_prediction, model = predict_with_glm(
                family_with_link,
                f(x_train),
                y_train,
                f(x_test),
            )
            r2s[col][name] = metrics.r2_score(y_test, y_prediction)
            models[col][name] = model
    return pd.DataFrame(r2s).T, models


def predict_with_glm(family_with_link, x_train, y_train, x_test, add_constant=True):
    if add_constant:
        x_train = sm.tools.add_constant(x_train, has_constant='add')
        x_test = sm.tools.add_constant(x_test, has_constant='add')
    y_glm = sm.GLM(y_train, x_train, family=family_with_link)
    y_model = y_glm.fit()
    return y_model.predict(x_test), y_model


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


def get_ages(conn, oas, year=2021):
    oas_str = '(' + ','.join(['"' + oa + '"' for oa in oas]) + ')'
    db_query = f'SELECT * FROM age_data WHERE year={year} and geography in {oas_str}'
    df = pd.read_sql(db_query, conn)
    return df.drop(['db_id', 'year'], axis=1).set_index('geography')


def select_house_data(conn, oas, distance, year):
    oas_str = '(' + ','.join(['"' + oa + '"' for oa in oas]) + ')'
    db_query = (f'SELECT * FROM oa_pp_data WHERE output_area in {oas_str} AND '
                f'year={year} AND distance={distance} '
                f'ORDER BY output_area')
    df = pd.read_sql(db_query, conn)
    return df.loc[:, ~df.columns.duplicated()].set_index('output_area').drop(['db_id', 'year'], axis=1)


def get_oa_house_data(conn, oas, distance=1000, year=2021):
    df = select_house_data(conn, oas, distance, year)
    oas_to_find = list(set(oas) - set(df.index.tolist()))
    if len(oas_to_find) > 0:
        access.insert_oa_house_data(conn, oas_to_find, distance, year)
        df = select_house_data(conn, oas, distance, year)
    return df


def plot_r2s_bar_chart(r2s):
    fig, ax = plt.subplots()
    r2s.plot(kind='bar', ax=ax)
    ax.set_title('R squared for a single feature with a constant, \n transformed by given functions')
    plt.show()


def plot_single_feature_predictors(rows, cols, train_x, all_y, models, design_fns, title):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*3 + 1), sharey=True)
    if rows > 1 and cols > 1:
        axes = list(np.concatenate(axes).flat)
    fig.suptitle(title)
    for i, col in enumerate(train_x.columns):
        axes[i].plot(train_x[col], all_y.loc[train_x.index], 'b.')
        axes[i].set_title(col)
        axes[i].set_ylim(2.5, 5)
        x = np.linspace(train_x[col].min(), train_x[col].max(), 100)
        for name, f in design_fns.items():
            y = models[col][name].predict(sm.tools.add_constant(f(x)))
            axes[i].plot(x, y, '-r')
    plt.show()


def plot_single_feature_residuals(rows, cols, test_x, all_y, models, design_fn_name='linear', design_fn=lambda x: x):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*3 + 1), sharey=True)
    if rows > 1 and cols > 1:
        axes = list(np.concatenate(axes).flat)
    fig.suptitle('Residuals')
    for i, col in enumerate(test_x.columns):
        axes[i].set_title(col)
        x = test_x[col]
        y = models[col][design_fn_name].predict(sm.tools.add_constant(design_fn(x)))
        axes[i].plot(x, all_y.loc[test_x.index] - y, 'b.')
    plt.show()


def get_coordinates_for_oas(conn, output_areas):
    oas_str = '(' + ','.join(['"' + oa + '"' for oa in output_areas]) + ')'
    db_query = (f'SELECT output_area, latitude, longitude '
                f'FROM oa_data '
                f'WHERE output_area in {oas_str} '
                f'ORDER BY output_area')
    return pd.read_sql(db_query, conn).set_index('output_area')
