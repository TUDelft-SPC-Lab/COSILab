#!/usr/bin/env python

from datetime import datetime as dt
from pathlib import Path

def parse_timestamps(timestamps, sensor_name):
    """
    Convert raw timestamps to datetime objects.
    
    Args:
        timestamps: List of raw timestamp values
        sensor_name: Name of the sensor for error reporting
    
    Returns:
        List of datetime objects or "Date error" strings for failed conversions
    """
    sensor_name = Path(sensor_name).stem
    error_reported = False
    timestamps_dt = []
    for x in timestamps:
        try:
            timestamps_dt.append(dt.fromtimestamp(float(x[0])/1000))
        except Exception as e:
            if error_reported is False:
                print('Error in timestamp conversion for sensor {}: {}'.format(sensor_name, str(e)))
                error_reported = True
            timestamps_dt.append("Date error")
    return timestamps_dt 