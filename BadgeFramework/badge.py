from __future__ import division, absolute_import, print_function
import time
import logging
import sys
import struct
import Queue

# Scan every 1 seconds, where each scan lasts for 0.25 seconds
DEFAULT_SCAN_WINDOW = 250
DEFAULT_SCAN_INTERVAL = 1000

DEFAULT_IMU_ACC_FSR = 4  # Valid ranges: 2, 4, 8, 16
DEFAULT_IMU_GYR_FSR = 1000  # Valid ranges: 250, 500, 1000, 2000
DEFAULT_IMU_DATARATE = 50

DEFAULT_MICROPHONE_MODE = 0	#Valid options: 0=Stereo, 1=Mono

from badge_protocol import *

logger = logging.getLogger(__name__)

# -- Helper methods used often in badge communication -- 

# We generally define timestamp_seconds to be in number of seconds since UTC epoch
# and timestamp_miliseconds to be the miliseconds portion of that UTC timestamp.

# Returns the current timestamp as two parts - seconds and milliseconds
def get_timestamps():
    return get_timestamps_from_time(time.time())

# Returns the given time as two parts - seconds and milliseconds
def get_timestamps_from_time(t):
    timestamp_seconds = int(t)
    timestamp_fraction_of_second = t - timestamp_seconds
    timestamp_ms = int(1000 * timestamp_fraction_of_second)
    return (timestamp_seconds, timestamp_ms)


# Convert badge timestamp representation to python representation
def timestamps_to_time(timestamp_seconds, timestamp_miliseconds):
    return float(timestamp_seconds) + (float(timestamp_miliseconds) / 1000.0)


# Represents an OpenBadge currently connected via the BadgeConnection 'connection'.
#    The 'connection' should already be connected when it is used to initialize this class.
# Implements methods that allow for interaction with that badge. 
class OpenBadge(object):
    def __init__(self, connection):
        self.connection = connection
        self.status_response_queue = Queue.Queue()
        self.start_microphone_response_queue = Queue.Queue()
        self.start_scan_response_queue = Queue.Queue()
        self.start_imu_response_queue = Queue.Queue()
        self.free_sdc_space_response_queue = Queue.Queue()
        self.sdc_errase_all_response_queue = Queue.Queue()
        self.get_imu_data_response_queue = Queue.Queue()
        self.get_fw_version_response_queue = Queue.Queue()
        self.list_files_response_queue = Queue.Queue()
        self.start_download_response_queue = Queue.Queue()
        self.download_chunk_response_queue = Queue.Queue()
        self.get_file_checksum_response_queue = Queue.Queue()
        

    # Helper function to send a BadgeMessage `command_message` to a device, expecting a response
    # of class `response_type` that is a subclass of BadgeMessage, or None if no response is expected.
    def send_command(self, command_message, response_type):
        expected_response_length = response_type.length() if response_type else 0

        serialized_command = command_message.serialize_message()
        logger.debug(
            "Sending: {}, Raw: {}".format(
                command_message, serialized_command.encode("hex")
            )
        )
        serialized_response = self.connection.send(
            serialized_command, response_len=expected_response_length
        )

        if expected_response_length > 0:
            response = response_type.deserialize_message(serialized_response)
            logger.info("Recieved response {}".format(response))
            return response
        else:
            logger.info("No response expected, transmission successful.")
            return True

    def send_request(self, request_message):
        # Clear the queue of any stale data before sending a new command
        if not self.connection.rx_queue.empty():
            with self.connection.rx_queue.mutex:
                self.connection.rx_queue.queue.clear()

        serialized_request = request_message.encode()

        # Adding length header:
        serialized_request_len = struct.pack("<H", len(serialized_request))
        serialized_request = serialized_request_len + serialized_request

        logger.debug(
            "Sending: {}, Raw: {}".format(
                request_message, serialized_request.encode("hex")
            )
        )

        self.connection.send(serialized_request, response_len=0)

    def receive_response(self):
        response_len = struct.unpack("<H", self.connection.await_data(2))[0]
        logger.debug("Wait response len: " + str(response_len))
        serialized_response = self.connection.await_data(response_len)

        response_message = Response.decode(serialized_response)

        queue_options = {
            Response_status_response_tag: self.status_response_queue,
            Response_start_microphone_response_tag: self.start_microphone_response_queue,
            Response_start_scan_response_tag: self.start_scan_response_queue,
            Response_start_imu_response_tag: self.start_imu_response_queue,
            Response_free_sdc_space_response_tag: self.free_sdc_space_response_queue,
            Response_sdc_errase_all_response_tag: self.sdc_errase_all_response_queue,
            Response_get_imu_data_response_tag: self.get_imu_data_response_queue,
            Response_get_fw_version_response_tag: self.get_fw_version_response_queue,
            Response_list_files_response_tag: self.list_files_response_queue,
            Response_start_download_response_tag: self.start_download_response_queue,
            Response_download_chunk_response_tag: self.download_chunk_response_queue,
            Response_get_file_checksum_response_tag: self.get_file_checksum_response_queue
        }
        response_options = {
            Response_status_response_tag: response_message.type.status_response,
            Response_start_microphone_response_tag: response_message.type.start_microphone_response,
            Response_start_scan_response_tag: response_message.type.start_scan_response,
            Response_start_imu_response_tag: response_message.type.start_imu_response,
            Response_free_sdc_space_response_tag: response_message.type.free_sdc_space_response,
            Response_sdc_errase_all_response_tag: response_message.type.sdc_errase_all_response,
            Response_get_imu_data_response_tag: response_message.type.get_imu_data_response,
            Response_get_fw_version_response_tag: response_message.type.get_fw_version_response,
            Response_list_files_response_tag: response_message.type.list_files_response,
            Response_start_download_response_tag: response_message.type.start_download_response,
            Response_download_chunk_response_tag: response_message.type.download_chunk_response,
            Response_get_file_checksum_response_tag: response_message.type.get_file_checksum_response
        }
        queue_options[response_message.type.which].put(
            response_options[response_message.type.which]
        )

    # Sends a status request to this Badge.
    #   Optional fields new_id and new_group number will set the badge's id
    #     and group number. They must be sent together.
    # Returns a StatusResponse() representing badge's response.
    def get_status(self, t=None, new_id=None, new_group_number=None):
        if t is None:
            (timestamp_seconds, timestamp_ms) = get_timestamps()
        else:
            (timestamp_seconds, timestamp_ms) = get_timestamps_from_time(t)

        request = Request()
        request.type.which = Request_status_request_tag
        request.type.status_request = StatusRequest()
        request.type.status_request.timestamp = Timestamp()
        request.type.status_request.timestamp.seconds = timestamp_seconds
        request.type.status_request.timestamp.ms = timestamp_ms
        if not ((new_id is None) or (new_group_number is None)):
            request.type.status_request.badge_assignement = BadgeAssignement()
            request.type.status_request.badge_assignement.ID = new_id
            request.type.status_request.badge_assignement.group = new_group_number
            request.type.status_request.has_badge_assignement = True

        self.send_request(request)

        # Clear the queue before receiving
        with self.status_response_queue.mutex:
            self.status_response_queue.queue.clear()

        while self.status_response_queue.empty():
            self.receive_response()

        return self.status_response_queue.get()

    # Sends a request to the badge to start recording microphone data.
    # Returns a StartRecordResponse() representing the badges response.
    def start_microphone(self, t=None, mode=DEFAULT_MICROPHONE_MODE):
        if t is None:
            (timestamp_seconds, timestamp_ms) = get_timestamps()
        else:
            (timestamp_seconds, timestamp_ms) = get_timestamps_from_time(t)
        #print("MODE ",mode)
        request = Request()
        request.type.which = Request_start_microphone_request_tag
        request.type.start_microphone_request = StartMicrophoneRequest()
        request.type.start_microphone_request.timestamp = Timestamp()
        request.type.start_microphone_request.timestamp.seconds = timestamp_seconds
        request.type.start_microphone_request.timestamp.ms = timestamp_ms
        request.type.start_microphone_request.mode = mode

        self.send_request(request)

        with self.start_microphone_response_queue.mutex:
            self.start_microphone_response_queue.queue.clear()

        while self.start_microphone_response_queue.empty():
            self.receive_response()

        return self.start_microphone_response_queue.get()

    # Sends a request to the badge to stop recording.
    # Returns True if request was successfuly sent.
    def stop_microphone(self):

        request = Request()
        request.type.which = Request_stop_microphone_request_tag
        request.type.stop_microphone_request = StopMicrophoneRequest()

        self.send_request(request)

    # Sends a request to the badge to start performing scans and collecting scan data.
    #   window_miliseconds and interval_miliseconds controls radio duty cycle during scanning (0 for firmware default)
    #     radio is active for [window_miliseconds] every [interval_miliseconds]
    # Returns a StartScanningResponse() representing the badge's response.
    def start_scan(
        self, t=None, window_ms=DEFAULT_SCAN_WINDOW, interval_ms=DEFAULT_SCAN_INTERVAL
    ):
        if t is None:
            (timestamp_seconds, timestamp_ms) = get_timestamps()
        else:
            (timestamp_seconds, timestamp_ms) = get_timestamps_from_time(t)

        request = Request()
        request.type.which = Request_start_scan_request_tag
        request.type.start_scan_request = StartScanRequest()
        request.type.start_scan_request.timestamp = Timestamp()
        request.type.start_scan_request.timestamp.seconds = timestamp_seconds
        request.type.start_scan_request.timestamp.ms = timestamp_ms
        request.type.start_scan_request.window = window_ms
        request.type.start_scan_request.interval = interval_ms

        self.send_request(request)

        # Clear the queue before receiving
        with self.start_scan_response_queue.mutex:
            self.start_scan_response_queue.queue.clear()

        while self.start_scan_response_queue.empty():
            self.receive_response()

        return self.start_scan_response_queue.get()

    # Sends a request to the badge to stop scanning.
    # Returns True if request was successfuly sent.
    def stop_scan(self):

        request = Request()
        request.type.which = Request_stop_scan_request_tag
        request.type.stop_scan_request = StopScanRequest()

        self.send_request(request)

    def start_imu(
        self,
        t=None,
        acc_fsr=DEFAULT_IMU_ACC_FSR,
        gyr_fsr=DEFAULT_IMU_GYR_FSR,
        datarate=DEFAULT_IMU_DATARATE,
    ):
        if t is None:
            (timestamp_seconds, timestamp_ms) = get_timestamps()
        else:
            (timestamp_seconds, timestamp_ms) = get_timestamps_from_time(t)

        request = Request()
        request.type.which = Request_start_imu_request_tag
        request.type.start_imu_request = StartImuRequest()
        request.type.start_imu_request.timestamp = Timestamp()
        request.type.start_imu_request.timestamp.seconds = timestamp_seconds
        request.type.start_imu_request.timestamp.ms = timestamp_ms
        request.type.start_imu_request.acc_fsr = acc_fsr
        request.type.start_imu_request.gyr_fsr = gyr_fsr
        request.type.start_imu_request.datarate = datarate

        self.send_request(request)

        # Clear the queue before receiving
        with self.start_imu_response_queue.mutex:
            self.start_imu_response_queue.queue.clear()

        while self.start_imu_response_queue.empty():
            self.receive_response()

        return self.start_imu_response_queue.get()

    def stop_imu(self):

        request = Request()
        request.type.which = Request_stop_imu_request_tag
        request.type.stop_imu_request = StopImuRequest()

        self.send_request(request)

    # Send a request to the badge to light an led to identify its self.
    #   If duration_seconds == 0, badge will turn off LED if currently lit.
    # Returns True if request was successfuly sent.
    def identify(self, duration_seconds=10):

        request = Request()
        request.type.which = Request_identify_request_tag
        request.type.identify_request = IdentifyRequest()
        request.type.identify_request.timeout = duration_seconds

        self.send_request(request)

        return True

    def restart(self):

        request = Request()
        request.type.which = Request_restart_request_tag
        request.type.restart_request = RestartRequest()

        self.send_request(request)

        return True

    def get_free_sdc_space(self):

        request = Request()
        request.type.which = Request_free_sdc_space_request_tag
        request.type.free_sdc_space_request = FreeSDCSpaceRequest()

        self.send_request(request)

        # Clear the queue before receiving
        with self.free_sdc_space_response_queue.mutex:
            self.free_sdc_space_response_queue.queue.clear()

        while self.free_sdc_space_response_queue.empty():
            self.receive_response()

        return self.free_sdc_space_response_queue.get()


    def sdc_errase_all(self):

        request = Request()
        request.type.which = Request_sdc_errase_all_request_tag
        request.type.sdc_errase_all_request = ErraseAllRequest()

        self.send_request(request)

        # Clear the queue before receiving
        with self.sdc_errase_all_response_queue.mutex:
            self.sdc_errase_all_response_queue.queue.clear()

        while self.sdc_errase_all_response_queue.empty():
            self.receive_response()

        return self.sdc_errase_all_response_queue.get()

    def get_imu_data(self):

        request = Request()
        request.type.which = Request_get_imu_data_request_tag
        request.type.get_imu_data_request = GetIMUDataRequest()
        request.type.get_imu_data_request.timestamp = Timestamp()

        self.send_request(request)

        # Clear the queue before receiving
        with self.get_imu_data_response_queue.mutex:
            self.get_imu_data_response_queue.queue.clear()

        while self.get_imu_data_response_queue.empty():
            self.receive_response()

        return self.get_imu_data_response_queue.get()
    
    def get_fw_version(self):

        request = Request()
        request.type.which = Request_get_fw_version_request_tag
        request.type.get_fw_version_request = GetFWVersionRequest()

        self.send_request(request)

        # Clear the queue before receiving
        with self.get_fw_version_response_queue.mutex:
            self.get_fw_version_response_queue.queue.clear()

        while self.get_fw_version_response_queue.empty():
            self.receive_response()

        version_response = self.get_fw_version_response_queue.get()
        version_response.version = version_response.version.replace("\x00", "")
        return version_response
        return self.get_fw_version_response_queue.get()
    
    def list_files(self, start_index=0):
        files = []
        current_start = start_index

        while True:
            request = Request()
            request.type.which = Request_list_files_request_tag
            request.type.list_files_request = ListFilesRequest()
            request.type.list_files_request.max_files = 3
            request.type.list_files_request.start_index = current_start

            self.send_request(request)

            with self.list_files_response_queue.mutex:
                self.list_files_response_queue.queue.clear()

            while self.list_files_response_queue.empty():
                self.receive_response()

            response = self.list_files_response_queue.get()

            for i in range(len(response.files)):
                clean_filename = response.files[i].filename.replace('\x00', '').strip()
                if clean_filename:
                    files.append({
                        'filename': clean_filename,
                        'size': response.files[i].file_size,
                        'timestamp': response.files[i].timestamp
                    })
            
            if (current_start + response.header.file_count) >= response.header.total_files:
                break

            current_start += response.header.file_count

        return files
    
    def download_file(self, filename, local_path=None, verify_checksum=True, show_progress=True):
        import os
        if local_path is None:
            local_path = filename

        # If file exists, verify checksum before downloading
        if os.path.exists(local_path) and verify_checksum:
            if show_progress:
                print("File {} already exists. Verifying checksum...".format(local_path))
            if self._verify_file_checksum(filename, local_path):
                if show_progress:
                    print("File already downloaded and checksum verified.")
                return True
            else:
                if show_progress:
                    print("Checksum mismatch. Re-downloading file.")

        request = Request()
        request.type.which = Request_start_download_request_tag
        request.type.start_download_request = StartDownloadRequest()
        request.type.start_download_request.filename = filename
        self.send_request(request)
        
        with self.start_download_response_queue.mutex:
            self.start_download_response_queue.queue.clear()
        
        while self.start_download_response_queue.empty():
            self.receive_response()
        start_response = self.start_download_response_queue.get()
        
        if not start_response.success:
            raise Exception("Failed to start download: {}".format(start_response.success))
        
        file_size = start_response.file_size
        total_chunks = start_response.total_chunks
        if show_progress:
            print("Downloading {} ({} chunks) to {}".format(filename, total_chunks, local_path))
        downloaded_data = bytearray()
        running_total = 0
        
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_chunks, desc="Downloading {}".format(filename), unit="chunk")
            except ImportError:
                print("tqdm not available, showing basic progress.")
        
        try:
            for chunk_index in range(total_chunks):
                chunk_request = Request()
                chunk_request.type.which = Request_download_chunk_request_tag
                chunk_request.type.download_chunk_request = DownloadChunkRequest()
                chunk_request.type.download_chunk_request.chunk_index = chunk_index
                self.send_request(chunk_request)
                
                # Clear queue and wait for response
                with self.download_chunk_response_queue.mutex:
                    self.download_chunk_response_queue.queue.clear()
                
                while self.download_chunk_response_queue.empty():
                    self.receive_response()
                
                chunk_response = self.download_chunk_response_queue.get()
                
                if chunk_response.chunk_size == 0:
                    raise Exception("Received empty chunk for index {}".format(chunk_index))
                
                # Convert list of integers to bytearray (Python 2 compatible)
                data_slice = chunk_response.data[:chunk_response.chunk_size]
                chunk_data = bytearray(data_slice)
                
                # Debug output
                running_total += len(chunk_data)
                print("DEBUG: Chunk {}: adding {} bytes, total: {}".format(
                    chunk_index, len(chunk_data), running_total))
                
                downloaded_data.extend(chunk_data)

                if progress_bar:
                    progress_bar.update(1)
                
                if chunk_response.is_last_chunk:
                    print("DEBUG: Last chunk received at index {}".format(chunk_index))
                    break
                    
        finally:
            if progress_bar:
                progress_bar.close()
        
        print("DEBUG: Final downloaded size: {}, expected: {}".format(len(downloaded_data), file_size))
        
        if len(downloaded_data) != file_size:
            raise Exception("Downloaded data size {} does not match expected size {}".format(len(downloaded_data), file_size))
        
        with open(local_path, 'wb') as f:
            f.write(downloaded_data)
        
        if verify_checksum:
            if show_progress:
                print("Verifying checksum...")
            if self._verify_file_checksum(filename, local_path):
                if show_progress:
                    print("Checksum verified")
            else:
                if show_progress:
                    print("Checksum verification failed")
                return False
        
        if show_progress:
            print("Successfully downloaded {} to {}".format(filename, local_path))
        return True

    def _verify_file_checksum(self, filename, local_path):
        request = Request()
        request.type.which = Request_get_file_checksum_request_tag
        request.type.get_file_checksum_request = GetFileChecksumRequest()
        request.type.get_file_checksum_request.filename = filename

        self.send_request(request)

        with self.get_file_checksum_response_queue.mutex:
            self.get_file_checksum_response_queue.queue.clear()

        while self.get_file_checksum_response_queue.empty():
            self.receive_response()

        response = self.get_file_checksum_response_queue.get()

        if not response.success:
            print("Failed to get checksum for {}: {}".format(filename, response.error_message))
            return False

        expected_checksum = response.checksum

        def calculate_crc32(data):
            crc = 0xFFFFFFFF
    
            for byte in bytearray(data):
                crc ^= byte
                for _ in range(8):
                    if crc & 1:
                        crc = (crc >> 1) ^ 0xEDB88320
                    else:
                        crc >>= 1
    
            return (~crc) & 0xFFFFFFFF  # Ensure unsigned 32-bit result
        
        with open(local_path, 'rb') as f:
            file_data = f.read()

        print("DEBUG: Expected checksum: {}, Actual checksum: {}".format(expected_checksum, calculate_crc32(file_data)))
        return calculate_crc32(file_data) == expected_checksum
    
    def download_all_files(self, output_dir="downloaded_data"):
        files = self.list_files()

        if not files:
            print("No files to download.")
            return {
                'success': 0,
                'failed': 0,
                'total': 0
            }

        import os
        # Python 2.7 compatible directory creation
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                # Directory might have been created by another process
                if not os.path.exists(output_dir):
                    print("Failed to create output directory: {}".format(e))
                    return {
                        'success': 0,
                        'failed': len(files),
                        'total': len(files)
                    }
        
        total_size = sum(f['size'] for f in files)
        print("Found {} files, total size: {:.1f} KB".format(len(files), total_size / 1024.0))

        success_count = 0
        failed_count = 0
        
        for file_info in files:
            filename = file_info['filename']
            
            # Clean filename for filesystem safety
            safe_filename = filename.replace('\x00', '').strip()
            if not safe_filename:
                print("Skipping file with invalid name")
                failed_count += 1
                continue
                
            local_path = os.path.join(output_dir, safe_filename)

            try:
                if self.download_file(safe_filename, local_path, show_progress=True):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print("Failed to download {}: {}".format(safe_filename, e))
                failed_count += 1

        result = {
            'success': success_count,
            'failed': failed_count,
            'total': len(files)
        }
        
        print("Downloaded {}/{} files successfully.".format(success_count, len(files)))
        return result
