#!/usr/bin/env python

import matplotlib.pyplot as plt
import struct
import numpy as np
import pandas as pd
from parser_utilities import parse_timestamps
import os
import fnmatch
from tqdm import tqdm

SENSOR_PREFIXES = ['ACC_', 'GYR_', 'MAG_', 'ROT_', 'SCAN_']

def is_sensor_file(filename):
    for prefix in SENSOR_PREFIXES:
        if filename.startswith(prefix) and not os.path.splitext(filename)[1]:
            return True
    return False

class IMUParser(object):
    def __init__(self, base_dir, output_dir=None):
        self.base_dir = os.path.abspath(base_dir)
        self.output_dir = os.path.abspath(output_dir) if output_dir else None
        self.accel_dfs = []
        self.gyro_dfs = []
        self.mag_dfs = []
        self.rot_dfs = []
        self.scan_dfs = []

    def _get_output_path(self, input_path, ext):
        if not self.output_dir:
            folder = os.path.dirname(input_path)
            basename = os.path.splitext(os.path.basename(input_path))[0]
            return os.path.join(folder, basename + ext)
        rel_path = os.path.relpath(input_path, self.base_dir)
        rel_folder = os.path.dirname(rel_path)
        basename = os.path.splitext(os.path.basename(input_path))[0]
        out_folder = os.path.join(self.output_dir, rel_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        return os.path.join(out_folder, basename + ext)

    def parse_and_store(self, file_path):
        basename = os.path.basename(file_path)
        if basename.startswith('ACC_'):
            df = self.parse_generic(file_path)
            self.accel_dfs.append({'df': df, 'source_file': file_path})
        elif basename.startswith('GYR_'):
            df = self.parse_generic(file_path)
            self.gyro_dfs.append({'df': df, 'source_file': file_path})
        elif basename.startswith('MAG_'):
            df = self.parse_generic(file_path)
            self.mag_dfs.append({'df': df, 'source_file': file_path})
        elif basename.startswith('ROT_'):
            df = self.parse_rot_file(file_path)
            self.rot_dfs.append({'df': df, 'source_file': file_path})
        elif basename.startswith('SCAN_'):
            df = self.parse_scan_file(file_path)
            self.scan_dfs.append({'df': df, 'source_file': file_path})

    def parse_generic(self,sensorname):
        data = []
        timestamps = []
        with open(sensorname, "rb") as f:
            byte = f.read()
            i=0
            while True:
                ts_bytes = byte[0+i:8+i]
                data_bytes = byte[8+i:20+i]
                if (len(data_bytes)) == 12 and (len(ts_bytes) == 8):
                    ts = struct.unpack('<Q',ts_bytes)
                    x,y,z = struct.unpack('<fff', data_bytes)
                    data.append([x,y,z])
                    timestamps.append(ts)
                    i = i + 24 # 8 timestamp + 12 data + 4 padding
                else:
                    break
        data_xyz = np.asarray(data)
        timestamps = np.asarray(timestamps)
        timestamps_dt = parse_timestamps(timestamps, sensorname)
        df = pd.DataFrame(timestamps_dt, columns=['time'])
        df['X'] = data_xyz[:,0]
        df['Y'] = data_xyz[:,1]
        df['Z'] = data_xyz[:,2]
        df.attrs['source_file'] = sensorname
        return df

    def parse_rot_file(self, rot_file):
        rotation = []
        timestamps = []
        with open(rot_file, "rb") as f:
            byte = f.read()
            i=0
            while True:
                ts_bytes = byte[0+i:8+i]
                rot_bytes = byte[8+i:24+i]
                if (len(rot_bytes)) == 16 and (len(ts_bytes) == 8):
                    ts = struct.unpack('<Q',ts_bytes)
                    q1,q2,q3,q4 = struct.unpack('<ffff', rot_bytes)
                    rotation.append([q1,q2,q3,q4])
                    timestamps.append(ts)
                    i = i + 24
                else:
                    break
        rotation_xyz = np.asarray(rotation)
        timestamps = np.asarray(timestamps)
        timestamps_dt = parse_timestamps(timestamps, rot_file)
        df = pd.DataFrame(timestamps_dt, columns=['time'])
        df['X'] = rotation_xyz[:,0]
        df['Y'] = rotation_xyz[:,1]
        df['Z'] = rotation_xyz[:,2]
        df['W'] = rotation_xyz[:,3]
        df.attrs['source_file'] = rot_file
        return df

    def parse_scan_file(self, scan_file):
        data = []
        timestamps = []
        with open(scan_file, "rb") as f:
            byte = f.read()
            i = 0
            while True:
                ts_bytes = byte[i : i + 8]
                data_bytes = byte[i + 8 : i + 16]
                if len(data_bytes) == 8 and len(ts_bytes) == 8:
                    ts = struct.unpack("<Q", ts_bytes)
                    sensor_id = struct.unpack("<H", data_bytes[0:2])[0]
                    group = struct.unpack("<b", data_bytes[2:3])[0]
                    rssi = struct.unpack("<b", data_bytes[3:4])[0]
                    data.append([sensor_id, rssi, group])
                    timestamps.append(ts)
                    i = i + 16
                else:
                    break
        data_xyz = np.asarray(data)
        timestamps = np.asarray(timestamps)
        timestamps_dt = parse_timestamps(timestamps, scan_file)
        df = pd.DataFrame(timestamps_dt, columns=['time'])
        if data_xyz.size > 0:
            df['SensorID'] = data_xyz[:, 0]
            df['RSSI'] = data_xyz[:, 1]
            df['Group'] = data_xyz[:, 2]
        df.attrs['source_file'] = scan_file
        return df

    def plot_dataframes(self, dfs, enabled):
        if enabled and dfs:
            for df_info in dfs:
                fname = self._get_output_path(df_info['source_file'], '.png')
                ax = df_info['df'].plot(x='time')
                fig = ax.get_figure()
                fig.savefig(fname)
                plt.close(fig)

    def plot_and_save(self, a, g, m, r):
        self.plot_dataframes(self.accel_dfs, a)
        self.plot_dataframes(self.gyro_dfs, g)
        self.plot_dataframes(self.mag_dfs, m)
        self.plot_dataframes(self.rot_dfs, r)

    def save_dataframes_generic(self, dfs, enabled):
        if enabled and dfs:
            for df_info in dfs:
                fname_base = self._get_output_path(df_info['source_file'], '')
                df_info['df'].to_pickle(fname_base + '.pkl')
                df_info['df'].to_csv(fname_base + '.csv')

    def save_dataframes(self,a,g,m,r,s):
        self.save_dataframes_generic(self.accel_dfs, a)
        self.save_dataframes_generic(self.gyro_dfs, g)
        self.save_dataframes_generic(self.mag_dfs, m)
        self.save_dataframes_generic(self.scan_dfs, s)
        self.save_dataframes_generic(self.rot_dfs, r)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(fn,acc,mag,gyr,rot,plot,scan,output_dir=None):
    parser = IMUParser(fn, output_dir=output_dir)
    # Recursively walk and process files
    for root_str, _, files in tqdm(list(os.walk(fn)), desc="Files"):
        root = os.path.abspath(root_str)
        for filename in files:
            if is_sensor_file(filename):
                file_path = os.path.join(root, filename)
                parser.parse_and_store(file_path)
    parser.save_dataframes(acc,mag,gyr,rot,scan)
    if plot:
        parser.plot_and_save(acc, mag, gyr, rot)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parser for the IMU data obtained from Minge Midges\
    (Acceleration, Gyroscope, Magnetometer, Rotation)')
    parser.add_argument('--fn', required=True,help='Please enter the path to the file')
    parser.add_argument('--acc', default=True,type=str2bool ,help='Check to parse and save acceleration data')
    parser.add_argument('--mag', default=True,type=str2bool ,help='Check to parse and save magnetometer data')
    parser.add_argument('--gyr', default=True,type=str2bool,help='Check to parse and save gyroscope data')
    parser.add_argument('--scan', default=True,type=str2bool,help='Check to parse and save scan data')
    parser.add_argument('--rot', default=True,type=str2bool ,help='Check to parse and save rotation data')
    parser.add_argument('--plot', default=True,type=str2bool ,help='Check to plot the parsed data')
    parser.add_argument('--output_dir', required=False, help='Optional output directory (will mirror input structure)')
    args = parser.parse_args()
    main(fn=args.fn, acc=args.acc, mag=args.mag, gyr=args.gyr, rot=args.rot, plot=args.plot, scan=args.scan, output_dir=args.output_dir)

    # Example command
    # python ./imu_parser.py --fn ../midge_0_files/ --scan TRUE --acc TRUE --mag TRUE --rot TRUE --gyr TRUE --rot TRUE --plot True
    # on windows, use python imu_parser.py --fn 'C:\\user\\midge_0_files\\' --scan ..., use single quotes and double backslashes
