#!/usr/bin/env python

from __future__ import division, absolute_import, print_function
import logging
import sys
import threading
import time

from badge import *
from ble_badge_connection import *
from bluepy import *
from bluepy import btle
from bluepy.btle import UUID, Peripheral, DefaultDelegate, AssignedNumbers ,Scanner
from bluepy.btle import BTLEException

# Enable debug output.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Main Loop of Badge Terminal

def main():
	num_args = len(sys.argv) # Get the arguments list
	if num_args != 2:
		print("Please enter badge MAC address")
		return

	device_addr = sys.argv[1]
	print("Connecting to badge", device_addr)
	connection = BLEBadgeConnection.get_connection_to_badge(device_addr)

	if not connection:
		print("Could not find nearby OpenBadge. :( Please try again!")
		return

	connection.connect()
	badge = OpenBadge(connection)

	print("Connected!")

	def print_help(args):
		print("Available commands: [optional arguments]")
		print("  status [new badge id] [group number] (id + group must be set together)")
		print("  start_microphone")
		print("  stop_microphone")
		print("  start_scan")
		print("  stop_scan")
		print("  start_imu")
		print("  stop_imu")
		print("  identify [led duration seconds | 'off']")
		print("  restart")
		print("  get_free_space")
		print("  sdc_errase_all")
		print("  get_imu_data")
		print("  list_files")
		print("  download_file [filename] [output_path]")
		print("  download_all [output_directory]")
		print("  help")
		print("All commands use current system time as transmitted time.")
		print("Default arguments used where not specified.")

	def handle_status_request(args):
		if len(args) == 1:
			print(badge.get_status())
		elif len(args) == 2:
			print("Badge ID and Group Number Must Be Set Simultaneously")
		elif len(args) == 3:
			new_id = int(args[1])
			group_number = int(args[2])
			print(badge.get_status(new_id=new_id, new_group_number=group_number))
		else:
			print("Invalid Syntax: status [new badge id] [group number]")

	def handle_start_microphone_request(args):
		print(badge.start_microphone(t=None,mode=0))

	def handle_stop_microphone_request(args):
		badge.stop_microphone()


	def handle_start_scan_request(args):
		print(badge.start_scan())

	def handle_stop_scan_request(args):
		badge.stop_scan()


	def handle_start_imu_request(args):
		print(badge.start_imu())

	def handle_stop_imu_request(args):
		badge.stop_imu()


	def handle_identify_request(args):
		if len(args) == 1:
			badge.identify()
		elif len(args) == 2:
			if args[1] == "off":
				badge.identify(duration_seconds=0)
			else:
				badge.identify(duration_seconds=int(args[1]))
		else:
			print("Invalid Syntax: identify [led duration seconds | 'off']")
			return

	def handle_restart_request(args):
		print(badge.restart())

	def handle_get_free_space(args):
		print(badge.get_free_sdc_space())

	def handle_sdc_errase_all(args):
		print(badge.sdc_errase_all())	

	def handle_get_imu_data(args):
		print(badge.get_imu_data())		

	def handle_list_files(args):
		try:
			print("Listing files on badge...")
			files = badge.list_files()
			
			if not files:
				print("No files found on badge")
				return
			
			total_size = sum(f['size'] for f in files)
			print("\nFound {} files, total size: {:.1f} KB".format(len(files), total_size / 1024.0))
			print("-" * 60)
			print("{:<25} {:<15} {:<15}".format('Filename', 'Size (KB)', 'Timestamp'))
			print("-" * 60)
			
			for file_info in files:
				size_kb = file_info['size'] / 1024.0
				timestamp = file_info.get('timestamp', 'N/A')
				print("{:<25} {:<15.1f} {:<15}".format(file_info['filename'], size_kb, timestamp))
			
			print("-" * 60)
			print("Total: {} files, {:.1f} KB".format(len(files), total_size / 1024.0))
			
		except Exception as e:
			print("Error listing files: {}".format(e))

	def handle_download_file(args):
		if len(args) < 2:
			print("Usage: download_file [filename] [output_path]")
			return
		
		filename = args[1]
		output_path = args[2] if len(args) > 2 else filename

		try:
			success = badge.download_file(filename, output_path, verify_checksum=True)
			if success:
				print("File '{}' downloaded successfully to '{}'.".format(filename, output_path))
			else:
				print("Failed to download file '{}'.".format(filename))
		except Exception as e:
			print("Error downloading file '{}': {}".format(filename, e))

	def handle_download_all(args):
		output_dir = args[1] if len(args) > 1 else "downloaded_files"

		try:
			result = badge.download_all_files(output_dir)
			if result['success'] > 0:
				print("\nDownload Summary:")
				print(" - Successfully downloaded: {}/{} files".format(result['success'], result['total']))
				if result['failed'] > 0:
					print(" - Failed files: {}".format(result['failed']))

				erase_response = input("\nErase SD card after successful download? (y/n): ")
				if erase_response.lower() == 'y':
					print ("Erasing SD card...")
					badge.sdc_errase_all()
					print("SD card erased successfully.")
			else:
				print("No files to download or all downloads failed.")
		except Exception as e:
			print("Error downloading files: {}".format(e))


	command_handlers = {
		"help": print_help,
		"status": handle_status_request,
		"start_microphone": handle_start_microphone_request,
		"stop_microphone": handle_stop_microphone_request,
		"start_scan": handle_start_scan_request,
		"stop_scan": handle_stop_scan_request,
		"start_imu": handle_start_imu_request,
		"stop_imu": handle_stop_imu_request,
		"identify": handle_identify_request,
		"restart": handle_restart_request,
		"get_free_space": handle_get_free_space,
		"sdc_errase_all": handle_sdc_errase_all,
		"get_imu_data": handle_get_imu_data,
		"list_files": handle_list_files,
		"download_file": handle_download_file,
		"download_all": handle_download_all,
	}

	while True:
		sys.stdout.write("> ")
		# [:-1] removes newline character
		command = sys.stdin.readline()[:-1]
		if command == "exit":
			connection.disconnect()
			break

		command_args = command.split(" ")
		if command_args[0] in command_handlers:
			command_handlers[command_args[0]](command_args)
		else:
			print("Command Not Found!")
			print_help({})

if __name__ == "__main__":
	main()
