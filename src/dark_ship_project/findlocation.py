from read_AIS_data import read_AIS_data
import pandas as pd


def ships_full_rows_in_area_time(lat_target, lon_target, ais_folder, start_time, end_time, delta_deg=0.01):
    """
    Return full AIS rows for ships near a coordinate within a time window.

    Parameters:
    - lat_target, lon_target: target coordinates in degrees
    - ais_folder: folder containing AIS data
    - start_time, end_time: strings or datetime objects defining the time window
    - delta_deg: box size in degrees (~0.01 ≈ 1 km)

    Returns:
    - DataFrame of full AIS rows in the area and time window
    """
    # Read AIS data
    ais_df = read_AIS_data(ais_folder)

    # Ensure timestamp is datetime
    ais_df['timestamp'] = pd.to_datetime(ais_df['timestamp'])

    # Convert time window to datetime if needed
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Filter ships in bounding box
    area_df = ais_df[
        (ais_df['latitude'] >= lat_target - delta_deg) &
        (ais_df['latitude'] <= lat_target + delta_deg) &
        (ais_df['longitude'] >= lon_target - delta_deg) &
        (ais_df['longitude'] <= lon_target + delta_deg)
    ]

    # Filter ships in the time window
    area_time_df = area_df[
        (area_df['timestamp'] >= start_time) &
        (area_df['timestamp'] <= end_time)
    ]

    return area_time_df


# Example usage
if __name__ == "__main__":
    target_lat = 55.585277
    target_lon = -5.108056
    ais_folder = "202306"
    start_time = "2023-06-12 00:00:00"
    end_time = "2023-06-12 08:00:00"

    ships_in_area = ships_full_rows_in_area_time(
        target_lat, target_lon, ais_folder, start_time, end_time)

    if not ships_in_area.empty:
        print("Full AIS rows for ships in the area between {} and {}: ".format(
            start_time, end_time))
        print(ships_in_area.to_string(index=False))
    else:
        print("No ships detected in the area between {} and {}.".format(
            start_time, end_time))
