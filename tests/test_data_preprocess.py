import unittest
import os
import sys
sys.path.append("../")
sys.path.append("../framework")
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import patch, call, create_autospec, MagicMock
from data_preprocess.data_preprocess import * 
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import pickle

class TestRemoveNonASCII(unittest.TestCase):
    def test_remove_non_ascii(self):
        # Test with a string containing ASCII and non-ASCII characters
        input_str = "Hello, 你好, こんにちは"
        cleaned_str = remove_non_ascii(input_str)

        # Check that non-ASCII characters are removed
        self.assertEqual(cleaned_str, "Hello, , ")

        # Test with a string containing only ASCII characters
        input_str_ascii_only = "Hello, World!"
        cleaned_str_ascii_only = remove_non_ascii(input_str_ascii_only)

        # Check that the output is the same as the input for ASCII-only strings
        self.assertEqual(cleaned_str_ascii_only, input_str_ascii_only)


class TestReplaceNum(unittest.TestCase):
    def test_replace_num(self):
        # Test with a string containing numerical digits
        input_str = "There are 3 apples and 5 oranges."
        replaced_str = replace_num(input_str)

        # Check that numerical digits are replaced by the special token
        self.assertEqual(replaced_str, "There are [NUM] apples and [NUM] oranges.")

        # Test with a string containing no numerical digits
        input_str_no_digits = "No numbers here!"
        replaced_str_no_digits = replace_num(input_str_no_digits)

        # Check that the output is the same as the input for strings with no numerical digits
        self.assertEqual(replaced_str_no_digits, input_str_no_digits)


class TestReadFileContents(unittest.TestCase):
    def test_read_file_contents(self):
        # Create a temporary directory for testing
        with TemporaryDirectory() as temp_dir:
            # Create a sample log file for testing
            log_file_path = os.path.join(temp_dir, "sample_log_file_BGL.log")
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write("-Line 1\nLine 2\n-Line 3\nLine 4\nLine 5\n")

            # Call the function with the sample log file
            log_lines = read_file_contents(file=log_file_path, max_log_lines=4, save_path=temp_dir, verbose=False)

            # Check that the correct number of lines are read
            self.assertEqual(len(log_lines), 4)

            # Check that the 'labels' subdirectory is created
            labels_directory = os.path.join(temp_dir, "labels")
            self.assertTrue(os.path.exists(labels_directory))
            
            # Check that the labels file is created and contains the correct data
            labels_file_path = os.path.join(labels_directory, "sample_log_file_BGL_labels.pkl")
            self.assertTrue(os.path.exists(labels_file_path))
            with open(labels_file_path, "rb") as labels_file:
                labels = pickle.load(labels_file)
            self.assertEqual(labels, [1, 0, 1, 0])


class TestTrainTokenizer(unittest.TestCase):
    def test_train_tokenizer(self):
        # Create a sample DataFrame for testing
        data = {'log_text': ["This is log text 1.", "Another log text.", "Log text with [NUM] and special characters!"]}
        data_df = pd.DataFrame(data)

        # Create a temporary directory for testing
        with TemporaryDirectory() as temp_dir:
            # Call the function with the sample DataFrame
            trained_tokenizer = train_tokenizer(vocab_size=50, save_folder=temp_dir, data_df=data_df)

            # Check that the tokenizer is an instance of the Tokenizer class
            self.assertIsInstance(trained_tokenizer, Tokenizer)

            # Check that the tokenizer file is saved
            tokenizer_file_path = os.path.join(temp_dir, "tokenizer-trained.json")
            self.assertTrue(os.path.exists(tokenizer_file_path))

            # Check that the vocabulary size matches the specified size
            self.assertEqual(trained_tokenizer.get_vocab_size(), 50)

            # Check that special tokens are present in the vocabulary
            special_tokens = ["[CLS]", "[SEP]", "[NUM]", "[UNK]"]
            for token in special_tokens:
                self.assertTrue(token in trained_tokenizer.get_vocab())


class TestGetDataToTokenize(unittest.TestCase):
    def test_get_data_to_tokenize(self):
        # Create a temporary directory for testing
        with TemporaryDirectory() as temp_dir:
            # Create sample log files and directories for testing
            bgl_dir = os.path.join(temp_dir, "BGL")
            os.makedirs(bgl_dir)
            ssh_dir = os.path.join(temp_dir, "SSH")
            os.makedirs(ssh_dir)
            test_dir = os.path.join(temp_dir, "TEST")
            os.makedirs(test_dir)
            other_dir = os.path.join(temp_dir, "Other")
            os.makedirs(other_dir)

            bgl_log_file = os.path.join(bgl_dir, "bgl_log.log")
            with open(bgl_log_file, 'w') as log_file:
                log_file.write("BGL log line 1\nBGL log line 2\n")

            ssh_log_file = os.path.join(ssh_dir, "ssh_log.log")
            with open(ssh_log_file, 'w') as log_file:
                log_file.write("SSH log line 1\nSSH log line 2\n")

            test_log_file = os.path.join(test_dir, "test_log.log")
            with open(test_log_file, 'w') as log_file:
                log_file.write("test log line 1\ntest log line 2\ntest log line 2\ntest log line 2\n")

            # Create a log file in the 'Other' directory
            other_log_file = os.path.join(other_dir, "other_log.log")
            with open(other_log_file, 'w') as log_file:
                log_file.write("Other log line 1\nOther log line 2\nOther log line 2\n")

            # Call the function with the temporary directory
            data_to_tokenize = get_data_to_tokenize(data_folder=temp_dir, max_log_lines=3, save_path=temp_dir)

            # Check that log lines are correctly organized by log source
            self.assertEqual(len(data_to_tokenize["bgl_log"]), 2)
            self.assertEqual(len(data_to_tokenize["ssh_log"]), 2)
            self.assertEqual(len(data_to_tokenize["test_log"]), 3)
            self.assertEqual(len(data_to_tokenize["other_log"]), 3)





