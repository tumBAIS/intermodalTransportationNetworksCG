from shapely.geometry import Point
import geopandas as gp
# import osmnx as osm


def get_city_gdf(city='', state='', country='', epsg_code='epsg:31468'):

    # This is how the city gdf can be found, but this takes relatively long
    # city = osm.geocode_to_gdf({'city': city,
    #                            'state:': state,
    #                            'Country': country})
    #
    # return osm.project_gdf(city, to_crs=epsg_code)

    return gp.read_file('data/Munich.geojson')


def point_in_city(city_gdf, cords):
    geom = city_gdf.loc[0, 'geometry']
    return geom.intersects(Point(cords))

