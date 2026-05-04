#!/usr/bin/env python

import pandas as pd
import sys
from hub_utilities import (
    timeout_input,
    start_recording_all_devices,
    stop_recording_all_devices,
    synchronise_and_check_all_devices,
    erase_sdcard_all_devices,
    print_fw_version_all_devices,
    choose_function,
    Connection,
    clear_input_line,
)

def print_hub_commands(sync_frequency):
    print(" start_all: starts recording on all midges with all sensors")
    print(" stop_all: stops the recording on all midges")
    print(" erase_all: erase the recorded data on all midges")
    print(" fw_all: show the firmware version of all of the midges")
    print(" start_sync: trigger sync every %d seconds" % sync_frequency)
    print(" stop_sync: show the firmware version of all of the midges")
    print(" midge: connect to a single midge for individual management")
    print(" toggle_show_status: toggle whether to show the status of the midges after synchronisation")
    print(" exit: stop recording and exit the hub script")
    print(" help: prints this help message")
    sys.stdout.flush()

def start_recording_all_devices_with_sync(df, sync_frequency):
    start_recording_all_devices(df)
    answer = raw_input("Do you want to start synchronisation? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes", "Y", "Yes"):
        print("Synchronisation enabled every %d seconds." % sync_frequency)
        sys.stdout.flush()
        return True
    else:
        print("Synchronisation not started.")
        sys.stdout.flush()
        return False
    

if __name__ == "__main__":
    
    # How frequent the synchronisation is triggered, defaults to every 5 minutes
    # Any value between 0 seconds and 10 minutes is good according to
    # A Modular Approach for Synchronized Wireless Multimodal Multisensor Data Acquisition in Highly Dynamic Social Settings
    # https://dl.acm.org/doi/10.1145/3394171.3413697
    # on page page 3590, where it refers to 
    # An open-source sensor platform for analysis of group dynamics
    # https://arxiv.org/abs/1901.04977
    sync_frequency = 5*60
    
    show_status_on_sync = False # Show the status of the midge after synchronisation

    df = pd.read_csv('ingroup_exp.csv')
    df['Recording'] = None  # We do not know if the midges are recording or not before connecting to them
    df['Id_Set'] = None  # We do not know if the midges are recording or not before connecting to them

    do_synchronization = False
    print("Type help for a list commands")
    sys.stdout.flush()

    def ti_input(prompt=''):
        clear_input_line()
        return timeout_input(timeout=sync_frequency, prompt=prompt)

    while True:
        command = ti_input(prompt='> ')

        if command == "start_all":
            do_synchronization = start_recording_all_devices_with_sync(df, sync_frequency)
        elif command == "stop_all":
            do_synchronization = False
            stop_recording_all_devices(df)
        elif command == "erase_all":
            erase_sdcard_all_devices(df)
        elif command == "fw_all":
            print_fw_version_all_devices(df)
        elif command == "help":
            print_hub_commands(sync_frequency)
        elif command == "midge":
            print('Type the id of the Midge you want to connect or exit.')
            sys.stdout.flush()

            while True:
                command = ti_input(prompt='Midge Connection > ')

                if command == "":
                    if do_synchronization is True:
                        synchronise_and_check_all_devices(df, show_status=show_status_on_sync)
                elif command == "exit":
                    print("Exited single midge management")
                    sys.stdout.flush()
                    break
                else:
                    try:
                        midge_id = int(command)
                        current_mac_addr = (df.loc[df['Participant Id'] == midge_id]['Mac Address']).values[0]
                        cur_connection = Connection(midge_id, current_mac_addr)
                    except Exception as error:
                        print (str(error))
                        sys.stdout.flush()
                        continue
                    print ("Connected to the midge. For available commands, type help.")
                    while True:
                        command = ti_input(prompt='Midge: ' + str(midge_id) + ' > ')
                        
                        if command == "exit":
                            cur_connection.disconnect()
                            print("Midge disconnected")
                            sys.stdout.flush()
                            break
                        elif command == "":
                            if do_synchronization is True:
                                synchronise_and_check_all_devices(df,
                                                                  skip_id=midge_id,
                                                                  conn_skip_id=cur_connection,
                                                                  show_status=show_status_on_sync)
                        elif command != "":
                            try:
                                command_args = command.split(" ")
                                out = choose_function(cur_connection, command_args[0])
                                if out is not None:
                                    print (out)
                                    sys.stdout.flush()
                            except Exception as error:
                                print (str(error))
                                print("Command not found!")
                                sys.stdout.flush()
                                cur_connection.print_help()
                                continue
        elif command == "toggle_show_status":
            show_status_on_sync = not show_status_on_sync
            print("Show status on synchronisation: " + str(show_status_on_sync))
            sys.stdout.flush()
        elif command == "start_sync":            
            if do_synchronization is True:
                print("Synchronisation already enabled.")
                sys.stdout.flush()
            else:
                print("Synchronisation enabled every %d seconds." % sync_frequency)
                do_synchronization = True
        elif command == "stop_sync":
            if do_synchronization is False:
                print("Synchronisation already disabled.")
                sys.stdout.flush()
            else:
                print("Stopping synchronisation.")
                do_synchronization = False
        elif command == "exit":
            print("Exit hub script.")
            sys.stdout.flush()
            do_synchronization = False
            answer = raw_input("Do you want to stop recording? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes", "Y", "Yes"):
                stop_recording_all_devices(df)
            exit()
        elif command == "":
            if do_synchronization is True:
                synchronise_and_check_all_devices(df, show_status=show_status_on_sync)
        else:
            print('Unknown command \'%s\'. Type help for a list of valid commands.' % command)
            sys.stdout.flush()
