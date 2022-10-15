import Generator.geodata as gd
import pandas as pd


def get_trips_in_time(file, earliest_departure=0, latest_departure=1440):
    trips = file[(file['departure_time'] >= earliest_departure) & (file['departure_time'] <= latest_departure)]
    return trips


def get_trips_by_mode(file, modes=[]):
    trips = file[file['mode'].isin(modes)]
    return trips


def get_trips_in_city(file, city):
    nodes_origin = list(zip(file['originX'], file['originY']))
    nodes_destination = list(zip(file['destinationX'], file['destinationY']))
    ids = list(file['id'])

    print('Check every trip')
    trips_in_city = []
    for i in range(len(nodes_origin)):
        if gd.point_in_city(city, nodes_origin[i]) and gd.point_in_city(city, nodes_destination[i]):
            trips_in_city.append(i)

    print('Build trip file')
    trips = file.iloc[trips_in_city]
    return trips

def get_random_subset(file, subset, seed):
    return file.sample(frac=subset, random_state=seed)


if __name__ == '__main__':
    city = gd.get_city_gdf('Munich')
    file_csv = pd.read_csv('data/trips.csv')
    file = get_trips_in_city(file_csv, city)
    print('hello')