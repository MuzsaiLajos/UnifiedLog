import unittest
import sys
sys.path.append("../")
sys.path.append("../framework/")
import torch
import pickle
from framework.run import *

class TestDictToNeptuneLoggable(unittest.TestCase):
    def test_dict_to_neptune_loggable(self):
        # Test case with integer and boolean values
        input_dict = {'key1': 123, 'key2': {'nested_key': True}}
        expected_output_dict = {'key1': '123', 'key2': {'nested_key': 'True'}}

        result = dict_to_neptune_loggable(input_dict)

        self.assertEqual(result, expected_output_dict)

    def test_dict_to_neptune_loggable_with_strings(self):
        # Test case with string values, no conversion needed
        input_dict = {'key1': 'value1', 'key2': {'nested_key': 'value2'}}
        expected_output_dict = {'key1': 'value1', 'key2': {'nested_key': 'value2'}}

        result = dict_to_neptune_loggable(input_dict)

        self.assertEqual(result, expected_output_dict)

    def test_dict_to_neptune_loggable_with_mixed_types(self):
        # Test case with mixed types (int, string, boolean)
        input_dict = {'key1': 123, 'key2': 'value2', 'key3': {'nested_key': True}}
        expected_output_dict = {'key1': '123', 'key2': 'value2', 'key3': {'nested_key': 'True'}}

        result = dict_to_neptune_loggable(input_dict)

        self.assertEqual(result, expected_output_dict)

    def test_dict_to_neptune_loggable_empty_dict(self):
        # Test case with an empty dictionary
        input_dict = {}
        expected_output_dict = {}

        result = dict_to_neptune_loggable(input_dict)

        self.assertEqual(result, expected_output_dict)

    def test_dict_to_neptune_loggable_nested_empty_dict(self):
        # Test case with nested empty dictionaries
        input_dict = {'key1': {}, 'key2': {'nested_key': {}}}
        expected_output_dict = {'key1': {}, 'key2': {'nested_key': {}}}

        result = dict_to_neptune_loggable(input_dict)

        self.assertEqual(result, expected_output_dict)


class TestGenerateOpenstackLabels(unittest.TestCase):
    def test_generate_openstack_labels(self):
        # Test case with normal and abnormal log names
        train_data_dict = {
            'openstack_abnormal_1': ['log line 1', 'log line 2'],
            'openstack_normal_1': ['log line 1', 'log line 2'],
            'other_log': ['log line 1', 'log line 2'],
        }
        train_labels_dict = {}

        generate_openstack_labels(train_data_dict, train_labels_dict)

        expected_labels = {
            'openstack_abnormal_1': torch.ones(2),
            'openstack_normal_1': torch.zeros(2),
            'other_log': torch.tensor([]),  # No label for non-OpenStack log
        }

        for key in train_labels_dict.keys():
            self.assertTrue(torch.equal(train_labels_dict[key], expected_labels[key]))

    def test_generate_openstack_labels_empty_input(self):
        # Test case with an empty input dictionary
        train_data_dict = {}
        train_labels_dict = {}

        generate_openstack_labels(train_data_dict, train_labels_dict)

        self.assertEqual(train_labels_dict, {})

    def test_generate_openstack_labels_no_openstack_logs(self):
        # Test case with logs not containing "openstack" in the name
        train_data_dict = {
            'other_log_1': ['log line 1', 'log line 2'],
            'other_log_2': ['log line 1', 'log line 2'],
        }
        train_labels_dict = {}

        generate_openstack_labels(train_data_dict, train_labels_dict)

        for key in train_labels_dict.keys():
            self.assertEqual(train_labels_dict[key], torch.tensor([]))  # No label for non-OpenStack logs


class TestLoadTrainDataDetector(unittest.TestCase):
    def test_load_train_data_detector(self):
        # Test case with mock data and paths
        detector_params = {
            'anomaly_detector': {
                'train_paths': ['/path/to/log_data.pkl'],
            },
        }
        mock_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]


        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps(mock_data))):
            result = load_train_data_detector(detector_params)

        expected_result = {
            'log_data': [torch.tensor([1, 2, 3]).to("cuda"), torch.tensor([4, 5, 6]).to("cuda")],
        }

        for i in range(3):
            for j in range(2):
                self.assertEqual(result['log_data'][j][i], expected_result['log_data'][j][i])


class TestLoadTrainLabelsDetector(unittest.TestCase):
    def test_load_train_labels_detector(self):
        # Test case with mock data and paths
        detector_params = {
            'anomaly_detector': {
                'train_paths': ['/path/to/log_data.pkl'],
                'label_paths': ['/path/to/labels.pkl'],
            },
        }
        mock_labels = torch.tensor([0, 1, 0, 1])

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps(mock_labels))):
            detector_train_data = {'log_data_1': {}}
            result = load_train_labels_detector(detector_train_data, detector_params)

        expected_result = {
            'log_data_1': torch.tensor([0, 1, 0, 1]),
        }

        torch.testing.assert_close(result['log_data'], expected_result['log_data_1'])


class TestLoadTestDataDetector(unittest.TestCase):
    def test_load_test_data_detector(self):
        # Test case with mock data and paths
        detector_params = {
            'anomaly_detector': {
                'test_data_paths': ['/path/to/test_data_1.pkl'],
            },
        }
        mock_data_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps(mock_data_1))):
            result = load_test_data_detector(detector_params)

        expected_result = {
            'test_data_1': [torch.tensor([1, 2, 3]).to("cuda"), torch.tensor([4, 5, 6]).to("cuda"), torch.tensor([7, 8, 9]).to("cuda")],
        }

        torch.testing.assert_close(result['test_data_1'], expected_result['test_data_1'])

            
class TestLoadTestLabelsDetector(unittest.TestCase):
    def test_load_test_labels_detector(self):
        # Test case with mock data and paths
        detector_params = {
            'anomaly_detector': {
                'test_labels': ['/path/to/test_labels.pkl'],
                'test_data_paths': ['/path/to/test_data.pkl']
            },
        }
        mock_labels = torch.tensor([0, 1, 0])

        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=pickle.dumps(mock_labels))):
            result = load_test_labels_detector(detector_params)

        expected_result = {
            'test_data': torch.tensor([0, 1, 0]),
        }

        for key in expected_result.keys():
            torch.testing.assert_close(result[key], expected_result[key])