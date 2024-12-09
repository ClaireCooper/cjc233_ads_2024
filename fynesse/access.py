import csv
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import osmium as osm
import osmnx as ox
import osmnx.utils_geo
import pandas as pd
import pymysql
import requests
import shapely
from shapely.geometry import shape

from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with 
outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond 
the legal side also think about the ethical issues around this data."""


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


def hello_world():
    print("Hello from the data science library!")


def download_price_paid_data(year_from, year_to):
    """Download UK house price data for given year range.
    :param year_from: first year to include
    :param year_to: last year to include
    """
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to + 1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(conn, year):
    """ Selects housing and location data from the price paid and geo datasets joined on postcode for a given year and
        uploads to the prices_coordinates_data table.
    :param conn: database connection object
    :param year: year of data to upload
    """
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"
    with conn.cursor() as cur:
        print('Selecting data for year: ' + str(year))
        cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, '
                    f'pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, '
                    f'po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, '
                    f'tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer '
                    f'BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po '
                                                                       'ON pp.postcode = po.postcode')
        rows = cur.fetchall()

        csv_file_path = 'output_file.csv'

        # Write the rows to the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write the data rows
            csv_writer.writerows(rows)
        print('Storing data for year: ' + str(year))
        cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS "
                                                                  "TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' "
                                                                  "LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))


def buildings_in_area(latitude, longitude, box_side_length_m):
    bbox = osmnx.utils_geo.bbox_from_point((latitude, longitude), box_side_length_m / 2)
    return ox.features_from_bbox(bbox=bbox, tags={'building': True})


def houses_within_distance_from_point(conn, latitude, longitude, box_side_length_m, year):
    (north, south, east, west) = osmnx.utils_geo.bbox_from_point((latitude, longitude), box_side_length_m / 2)
    db_query = (f'SELECT * FROM (SELECT * FROM postcode_data WHERE latitude BETWEEN {south} AND {north} AND longitude '
                f'BETWEEN {west} AND {east}) AS po INNER JOIN (SELECT * FROM pp_data WHERE year(date_of_transfer) >= \''
                f'{year}\') AS pp ON po.postcode = pp.postcode')
    df = pd.read_sql(db_query, conn)
    return df.loc[:, ~df.columns.duplicated()]


def download_country_border():
    if not Path(f"./GBR.gpkg").is_file():
        url = (f"https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items"
               f"/31a5004908f749cbbce606d4c6d560b9/geoPackage?layers=0")
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"./GBR.gpkg", "wb") as f:
                f.write(response.content)


def download_census_data(code, base_dir=''):
    url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")


def load_census_data(code, level='msoa'):
    return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')


def select_all_from_table(conn, table):
    db_query = f'SELECT * FROM {table}'
    df = pd.read_sql(db_query, conn)
    return df.loc[:, ~df.columns.duplicated()]


def select_all_from_oa_table(conn):
    db_query = f'SELECT output_area, latitude, longitude, ST_AsBinary(geometry) as geometry FROM oa_data'
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:27700')
    return gdf.loc[:, ~df.columns.duplicated()]


def download_output_area_data():
    url = ('https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/6beafcfd9b9c4c9993a06b6b199d7e6d'
           '/geojson?layers=0')
    if not Path(f"./output_areas.geojson").is_file():
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"./output_areas.geojson", "wb") as f:
                f.write(response.content)


def download_country_osm_data(country, continent='europe'):
    url = (f"https://download.openstreetmap.fr/extracts/{continent.lower().replace(' ', '_')}/"
           f"{country.lower().replace(' ', '_')}.osm.pbf")

    output_file = Path(f"./{continent.lower().replace(' ', '_')}"
                       f"/{country.lower().replace(' ', '_')}.osm.pbf")
    if not output_file.is_file():
        output_file.parent.mkdir(exist_ok=True, parents=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"./{continent.lower().replace(' ', '_')}/"
                      f"{country.lower().replace(' ', '_')}.osm.pbf", "wb") as f:
                f.write(response.content)


def _inner(row, points, oas):
    for i in range(len(points)):
        if shapely.within(points[i], row['geometry']):
            oas[i] = row['output_area']


def select_output_areas_from_locations(conn, points, points_crs='EPSG:4326'):
    db_query = f'SELECT output_area, ST_AsBinary(geometry) as geometry FROM oa_data'
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:27700')
    gdf = gdf.loc[:, ~df.columns.duplicated()]
    gdf = gdf.to_crs(points_crs)
    oas = [None for point in range(len(points))]
    gdf.apply(lambda row: _inner(row, points, oas), axis=1)
    return oas


def save_tag_locations_as_csv(osm_file_path, tag_list):
    class _TagLocationHandler(osm.SimpleHandler):
        def __init__(self, tags_list):
            osm.SimpleHandler.__init__(self)
            self.tag_locations = []
            self.tags = tags_list
            self.fab = osm.geom.WKTFactory()

        def tag_inventory(self, elem, wkt):
            center = shapely.centroid(shapely.from_wkt(wkt))
            for tag in elem.tags:
                if tag in self.tags:
                    self.tag_locations.append([center.x,
                                               center.y,
                                               tag.k,
                                               tag.v,
                                               f"{wkt}"])
            if len(self.tag_locations) % 1000 == 0:
                print(len(self.tag_locations), "locations found")

        def node(self, n):
            wkt = self.fab.create_point(n.location)
            self.tag_inventory(n, wkt)

        def way(self, w):
            if not w.is_closed():
                wkt = self.fab.create_linestring(w.nodes)
                self.tag_inventory(w, wkt)

        def area(self, a):
            try:
                if a.from_way():
                    wkt = 'POLYGON' + self.fab.create_multipolygon(a)[13:-1]
                else:
                    wkt = self.fab.create_multipolygon(a)
                self.tag_inventory(a, wkt)
            except:
                print('Skipping area', a.id)

    handler = _TagLocationHandler(tag_list)
    handler.apply_file(osm_file_path, locations=True,
                       filters=[osm.filter.EmptyTagFilter(),
                                osm.filter.TagFilter(*tag_list)])
    print('Writing data to CSV...')
    with open('tag_locations.csv', 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(handler.tag_locations)
    print('Done.')


def save_key_locations_as_csv(osm_file_path, key_list):
    class _KeyLocationHandler(osm.SimpleHandler):
        def __init__(self, keys_list):
            osm.SimpleHandler.__init__(self)
            self.key_locations = []
            self.keys = keys_list
            self.fab = osm.geom.WKTFactory()

        def tag_inventory(self, elem, wkt):
            center = shapely.centroid(shapely.from_wkt(wkt))
            for tag in elem.tags:
                if tag.k in self.keys:
                    self.key_locations.append([center.x,
                                               center.y,
                                               tag.k,
                                               tag.v,
                                               f"{wkt}"])
            if len(self.key_locations) % 1000 == 0:
                print(len(self.key_locations), "locations found")

        def node(self, n):
            wkt = self.fab.create_point(n.location)
            self.tag_inventory(n, wkt)

        def way(self, w):
            if not w.is_closed():
                wkt = self.fab.create_linestring(w.nodes)
                self.tag_inventory(w, wkt)

        def area(self, a):
            try:
                if a.from_way():
                    wkt = 'POLYGON' + self.fab.create_multipolygon(a)[13:-1]
                else:
                    wkt = self.fab.create_multipolygon(a)
                self.tag_inventory(a, wkt)
            except:
                print('Skipping area', a.id)

    handler = _KeyLocationHandler(key_list)
    handler.apply_file(osm_file_path, locations=True,
                       filters=[osm.filter.EmptyTagFilter(),
                                osm.filter.KeyFilter(*key_list)])
    print('Writing data to CSV...')
    with open('key_locations.csv', 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(handler.key_locations)
    print('Done.')


def upload_csv_to_db(conn, table, path):
    with conn.cursor() as cur:
        cur.execute(f'LOAD DATA LOCAL INFILE "{path}" INTO TABLE `{table}` '
                    f'FIELDS TERMINATED BY "," LINES STARTING BY "" TERMINATED BY "\n";')
    conn.commit()


def upload_csv_to_db_with_geometry(conn, table, path, geometry_column='geometry'):
    columns = _get_column_names(conn, table)
    columns.remove('db_id')
    if geometry_column in columns:
        columns_str = ('(' + ','.join(columns) + ')').replace(geometry_column, '@geom')
        with conn.cursor() as cur:
            query = (f'LOAD DATA LOCAL INFILE "{path}" INTO TABLE `{table}` '
                     f'FIELDS TERMINATED BY "," OPTIONALLY ENCLOSED by \'"\' '
                     f'LINES STARTING BY "" TERMINATED BY "\n"{columns_str}'
                     f'SET {geometry_column} = ST_GeomFromText(@geom);')
            print(query)
            cur.execute(query)
        conn.commit()
    else:
        print(geometry_column, 'not a column in table:', table)


def _get_column_names(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = "{table}"')
        names = cur.fetchall()
    return [name[0] for name in names]


def save_df_to_csv_for_db(df, filename):
    df.to_csv(filename, header=False, index=False, lineterminator='\n')


def census_upload_join_data(conn):
    with conn.cursor() as cur:
        print('Selecting data...')
        cur.execute(f'SELECT oa.output_area, oa.latitude, oa.longitude, ST_AsText(oa.geometry), sec.L15, '
                    f'pd.population_density FROM oa_data AS oa INNER JOIN sec_data AS sec ON oa.output_area = '
                    f'sec.geography INNER JOIN pd_data as pd ON oa.output_area = pd.geography')
        rows = cur.fetchall()

        csv_file_path = 'output_file.csv'
        print('Storing data to CSV...')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, lineterminator='\n')
            csv_writer.writerows(rows)
        print('Uploading data...')
        cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `census_oa_data` FIELDS TERMINATED BY "
                                                                  "',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY "
                                                                  "'' TERMINATED BY '\n' (output_area, latitude, "
                                                                  "longitude, @geom, L15, population_density) SET "
                                                                  "geometry = ST_GeomFromText(@geom);")
    conn.commit()
    print('Data uploaded.')


def select_all_from_table_with_geometry(conn, table, geometry_column='geometry', crs='EPSG:4326'):
    db_query = f'SELECT *, ST_AsBinary({geometry_column}) as geometry_bin FROM {table}'
    df = pd.read_sql(db_query, conn)
    gs = gpd.GeoSeries.from_wkb(df['geometry_bin'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs=crs)
    return gdf.loc[:, ~df.columns.duplicated()].drop('geometry_bin', axis=1)


def osm_in_oa_radius_counts_to_csv(conn, distance, year, tag, value):
    with conn.cursor() as cur:
        print('Selecting data...')
        cur.execute(f'select year, output_area, tagkey, tagvalue, {distance} as distance, count(*) as count from ('
                    f'select year, output_area, latitude, longitude from oa_data where year = {year}) as oa cross '
                    f'join (select latitude, longitude, tagkey, tagvalue from osm_data where tagkey="{tag}" and '
                    f'tagvalue="{value}") as osm where ST_DISTANCE_SPHERE(POINT(oa.longitude, oa.latitude), '
                    f'POINT(osm.longitude, osm.latitude)) < {distance} GROUP BY year, output_area, tagkey, tagvalue;')
        rows = cur.fetchall()

        csv_file_path = f'osm_oa_{tag}_{value}.csv'
        print('Storing data to CSV...')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, lineterminator='\n')
            csv_writer.writerows(rows)


def houses_sold_within_distance_box_from_point_in_year(conn, latitude, longitude, distance, year):
    start_date = str(year) + '-01-01'
    end_date = str(year) + '-12-31'
    (north, south, east, west) = osmnx.utils_geo.bbox_from_point((latitude, longitude), distance)
    db_query = (f'SELECT price, property_type, new_build_flag FROM (SELECT postcode FROM postcode_data '
                f'WHERE latitude BETWEEN {south} AND {north} AND longitude BETWEEN {west} AND {east}) AS po '
                f'INNER JOIN (SELECT * FROM pp_data WHERE date_of_transfer BETWEEN \'{start_date}\' AND \'{end_date}\' '
                f'AND tenure_type="F" AND property_type<>"O")'
                f'AS pp ON po.postcode = pp.postcode')
    df = pd.read_sql(db_query, conn)
    return df.loc[:, ~df.columns.duplicated()]


def get_coordinates_for_oa(conn, oa, year=2021):
    db_query = f'select latitude, longitude from oa_data where output_area = \'{oa}\' and year = {year};'
    df = pd.read_sql(db_query, conn)
    return df.loc[0, ~df.columns.duplicated()]


def save_oa_house_data_to_csv(conn, oas, distance, year):
    prop_type = ['F', 'T', 'S', 'D']
    rows = []
    for oa in oas:
        row = []
        coordinates = get_coordinates_for_oa(conn, oa)
        houses_df = houses_sold_within_distance_box_from_point_in_year(conn, coordinates.latitude,
                                                                       coordinates.longitude, distance, year)
        row += [year, oa, distance]
        row += [houses_df.price.max(), houses_df.price.min(), houses_df.price.mean().astype(int),
                houses_df.price.median()]
        row += [len(houses_df[houses_df.property_type == p]) for p in prop_type]
        row += [len(houses_df[houses_df.new_build_flag == 'Y']), len(houses_df.index)]
        rows.append(row)

    csv_file_path = f'pp_oa.csv'
    print('Storing data to CSV...')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator='\n')
        csv_writer.writerows(rows)


def insert_feature_count_for_output_area(conn, output_area, tagkey, tagvalue, count, distance=1000, year=2021):
    db_query = (
        f"INSERT INTO osm_oa_radius_counts (year, output_area, tagkey, tagvalue, distance, count) "
        f"VALUES({year}, '{output_area}', '{tagkey}', '{tagvalue}', {distance}, {count})")
    with conn.cursor() as cur:
        cur.execute(db_query)
    conn.commit()


def insert_oa_house_data(conn, oas, distance, year):
    prop_type = ['F', 'T', 'S', 'D']
    rows = []
    for oa in oas:
        row = []
        coordinates = get_coordinates_for_oa(conn, oa)
        houses_df = houses_sold_within_distance_box_from_point_in_year(conn, coordinates.latitude,
                                                                       coordinates.longitude, distance, year)
        row += [year, oa, distance]
        if not houses_df.empty:
            row += [houses_df.price.max(), houses_df.price.min(), int(houses_df.price.mean()),
                    int(houses_df.price.median())]
            row += [len(houses_df[houses_df.property_type == p]) for p in prop_type]
            row += [len(houses_df[houses_df.new_build_flag == 'Y']), len(houses_df.index)]
        else:
            row += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        rows.append(row)

    csv_file_path = f'pp_oa.csv'
    print('Storing data to CSV...')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, lineterminator='\n')
        csv_writer.writerows(rows)
    print('Uploading CSV to DB...')
    upload_csv_to_db(conn, 'oa_pp_data', csv_file_path)
