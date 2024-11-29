import csv
import io
import zipfile
from pathlib import Path

import osmnx as ox
import osmnx.utils_geo
import pandas as pd
import geopandas as gpd
import pymysql
import requests
import shapely

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
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT * FROM (SELECT * FROM postcode_data WHERE latitude BETWEEN {south} AND {north} AND longitude '
            f'BETWEEN {west} AND {east}) AS po INNER JOIN (SELECT * FROM pp_data WHERE year(date_of_transfer) >= \''
            f'{year}\') AS pp ON po.postcode = pp.postcode')
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    return df.loc[:, ~df.columns.duplicated()]


def download_country_border(country_code):
    if not Path(f"./{country_code}.gpkg").is_file():
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{country_code}.gpkg"
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"./{country_code}.gpkg", "wb") as f:
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
    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM {table}')
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    return df.loc[:, ~df.columns.duplicated()]


def select_all_from_oa_table(conn):
    with conn.cursor() as cur:
        cur.execute(f'SELECT output_area, latitude, longitude, ST_AsBinary(geometry) as geometry FROM oa_data')
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
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


def select_output_areas_from_locations(conn, points, points_crs='EPSG:4326'):
    with conn.cursor() as cur:
        cur.execute(f'SELECT output_area, ST_AsBinary(geometry) as geometry FROM oa_data')
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    gs = gpd.GeoSeries.from_wkb(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry=gs, crs='EPSG:27700')
    gdf = gdf.loc[:, ~df.columns.duplicated()]
    gdf = gdf.to_crs(points_crs)
    oas = []
    for point in points:
        oas.append(gdf[gdf.geometry.apply(
            lambda g: shapely.within(point, g))]['output_area'][0])
    return oas
