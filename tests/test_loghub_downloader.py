import unittest
import os
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
import sys
sys.path.append("../")
sys.path.append("../framework")
from data_preprocess.loghub_downloader import download_files, unzip_files

class TestDownloadFiles(unittest.TestCase):
    def test_download_files(self):
        # Create a temporary directory to use as the save folder
        with TemporaryDirectory() as temp_dir:
            download_links = ["https://example.com/file1.txt", "https://example.com/file2.jpg"]
            save_folder = temp_dir

            # Mocking urllib.request.urlretrieve to avoid actual network requests
            with patch('urllib.request.urlretrieve') as mock_urlretrieve:
                # Call the function with the mocked urllib.request.urlretrieve
                download_files(download_links, save_folder)

                # Check that urllib.request.urlretrieve was called for each link with the correct arguments
                expected_calls = [call(link, os.path.join(save_folder, link.split("/")[-1].split("?")[0])) for link in download_links]
                mock_urlretrieve.assert_has_calls(expected_calls, any_order=True)

    # Add more test cases if needed

class TestUnzipFiles(unittest.TestCase):
    def test_unzip_files(self):
        # Create a temporary directory to use as the save folder
        with TemporaryDirectory() as temp_dir:
            # Create a test ZIP file
            zip_file_path = os.path.join(temp_dir, "test_file.zip")
            with open(zip_file_path, 'w') as test_file:
                test_file.write("Test content")

            # Call the function with the mocked os.system
            with patch('os.system') as mock_system:
                unzip_files(temp_dir)

                # Check that os.system was called with the correct arguments for unzipping
                mock_system.assert_called_with(f"unzip {zip_file_path} -d {os.path.join(temp_dir, 'test_file')}")

                # Check that the ZIP file was removed
                self.assertFalse(os.path.exists(zip_file_path))