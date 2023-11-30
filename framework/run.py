import yaml
from yaml.loader import SafeLoader
import neptune as neptune
import os
import torch
import pickle
from transformer_encoder import TransformerEncoder
from anomaly_detector import AnomalyDetector
import copy
import argparse


def initialize_neptune():
    """
    Initializes Neptune logging for experiment tracking.

    Raises:
    - ValueError: If the environment variable NEPTUNE_API_TOKEN is not set.

    Returns:
    neptune.Run: An instance of the Neptune Run object for experiment tracking.

    Example:
    ```python
    run = initialize_neptune()
    ```

    Note:
    This function attempts to retrieve the Neptune API token from the environment variable NEPTUNE_API_TOKEN.
    If the API token is not found, a ValueError is raised.
    The function then initializes a Neptune Run object for experiment tracking and returns it.
    """
    print("Initializing neptune logging...")
    try:
        api_token = os.environ["NEPTUNE_API_TOKEN"]
    except:
        raise ValueError("NEPTUNE_API_TOKEN not set as environment variable")
    run = neptune.init_run(
        project="aielte/CyberML",
        api_token=api_token,
    ) 
    return run


def dict_to_neptune_loggable(dict):
    """
    Converts a dictionary to Neptune loggable format, converting non-string values to strings.

    Parameters:
    - dictionary (dict): The input dictionary.

    Returns:
    dict: The modified dictionary with non-string values converted to strings.

    Example:
    ```python
    data_dict = {'key1': 123, 'key2': {'nested_key': True}}
    neptune_loggable_dict = dict_to_neptune_loggable(data_dict)
    ```

    Note:
    This function recursively traverses the input dictionary and converts non-string values to strings.
    The modified dictionary is returned in a format suitable for logging with Neptune.
    """
    for key in dict.keys():
        try:
            dict_to_neptune_loggable(dict[key])
        except:
            dict[key] = str(dict[key])
    return dict


def handle_encoder_training(transformer_encoder, params):
    """
    Handles the training process for a transformer encoder.

    Parameters:
    - transformer_encoder (TransformerEncoder): The transformer encoder model to be trained.
    - params (dict): A dictionary containing training parameters.

    Returns:
    None

    Example:
    ```python
    transformer = TransformerEncoder(...)
    training_params = {
        "transformer_encoder": {
            "epochs": 10,
            "train_paths": ["path/to/train_data.pkl"],
            "max_train_data_size": 1000,
            "train_val_test_split": [0.8, 0.1, 0.1],
            "batch_size": 32,
            "lr": 0.001,
            "save_path": "path/to/save/models",
            "pad_token_id": 0,
            "save_every_epoch": 2
        }
    }
    handle_encoder_training(transformer, training_params)
    ```

    Note:
    This function loads training data for the transformer encoder from the specified paths, reduces the dataset size if configured,
    and then trains the transformer encoder model using the provided parameters.
    The training data is organized into a dictionary, where keys are log source names and values are lists of PyTorch tensors.
    The training process includes splitting the data into training, validation, and test sets, specifying batch size, learning rate,
    and saving the model at specified intervals during training.
    """
    if params["transformer_encoder"]["epochs"] > 0:
        print(f"Loading train data for encoder... ({params['transformer_encoder']['train_paths']})")
        if params["transformer_encoder"]["max_train_data_size"] is not None:
            print(f'Keeping the first {params["transformer_encoder"]["max_train_data_size"]} lines from each dataset.')
        transformer_encoder_train_data_dict = {}
        for i in range(len(params["transformer_encoder"]["train_paths"])):
            with open(params["transformer_encoder"]["train_paths"][i], "rb") as file:
                transformer_encoder_train_data = pickle.load(file)

            # Reduce lines of specified in config
            if params["transformer_encoder"]["max_train_data_size"] is not None:
                transformer_encoder_train_data = transformer_encoder_train_data[:int(params["transformer_encoder"]["max_train_data_size"])]

            log_name = ".".join(params["transformer_encoder"]["train_paths"][i].split("/")[-1].split(".")[:-1])
            transformer_encoder_train_data_dict[log_name] = [torch.Tensor(x).long() for x in transformer_encoder_train_data]

        transformer_encoder.train(
            data = transformer_encoder_train_data_dict, 
            train_val_test_split = params["transformer_encoder"]["train_val_test_split"], 
            batch_size = params["transformer_encoder"]["batch_size"], 
            lr = params["transformer_encoder"]["lr"], 
            save_path = params["transformer_encoder"]["save_path"], 
            epochs = params["transformer_encoder"]["epochs"],
            padding_value = params["transformer_encoder"]["pad_token_id"],
            save_every_epoch=params["transformer_encoder"]["save_every_epoch"]
        )


def generate_openstack_labels(anomaly_detector_train_data_dict, anomaly_detector_train_labels_dict):
    """
    Generates labels for OpenStack log data based on log names.

    Parameters:
    - anomaly_detector_train_data_dict (dict): A dictionary where keys are log names, and values are lists of log lines.
    - anomaly_detector_train_labels_dict (dict): A dictionary to store generated labels, where keys are log names, and values are torch tensors.

    Returns:
    None

    Example:
    ```python
    train_data_dict = {
        'openstack_abnormal_1': ['log line 1', 'log line 2', ...],
        'openstack_normal_1': ['log line 1', 'log line 2', ...],
        ...
    }
    train_labels_dict = {}
    generate_openstack_labels(train_data_dict, train_labels_dict)
    ```

    Note:
    This function generates binary labels for OpenStack log data based on the log names.
    If the log name contains "openstack" and "abnormal" is not part of the name, the corresponding label is set to 0 (normal).
    If "abnormal" is part of the name, the label is set to 1 (abnormal).
    The generated labels are stored in the provided dictionary.
    """
    for key in anomaly_detector_train_data_dict.keys():
        if "openstack" in key:
            if "abnormal" not in key:
                anomaly_detector_train_labels_dict[key] = torch.zeros(len(anomaly_detector_train_data_dict[key]))
            else:
                anomaly_detector_train_labels_dict[key] = torch.ones(len(anomaly_detector_train_data_dict[key]))


def load_train_data_detector(params):
    """
    Load train data for the anomaly detector from specified paths.

    Parameters:
    - params (dict): A dictionary containing configuration parameters, including train data paths.

    Returns:
    dict: A dictionary where keys are log names, and values are torch tensors or nested dictionaries of torch tensors.

    Example:
    ```python
    detector_params = {
        'anomaly_detector': {
            'train_paths': ['/path/to/log_data_1.pkl', '/path/to/log_data_2.pkl'],
            ...
        },
        ...
    }
    detector_train_data = load_train_data_detector(detector_params)
    ```

    Note:
    This function loads train data for the anomaly detector from specified pickle files.
    It organizes the data into a dictionary, where keys are log names derived from file names, and values are torch tensors or nested dictionaries of torch tensors.
    For log names containing "HDFS_1" or "hadoop" (case-insensitive), the function creates nested dictionaries to organize data according to their respective subcategories.
    The loaded data is moved to the CUDA device for GPU acceleration.
    """
    print(f'Loading train data for detector...({params["anomaly_detector"]["train_paths"]})')
    anomaly_detector_train_data_dict = {}
    for i in range(len(params["anomaly_detector"]["train_paths"])):
        with open(params["anomaly_detector"]["train_paths"][i], "rb") as file:
            anomaly_detector_train_data = pickle.load(file)
        log_name = ".".join(params["anomaly_detector"]["train_paths"][i].split("/")[-1].split(".")[:-1])
        if "HDFS_1" in log_name or "hadoop" in log_name.lower():
            anomaly_detector_train_data_dict[log_name] = {}
            for key in anomaly_detector_train_data.keys():
                anomaly_detector_train_data_dict[log_name][key] = [torch.Tensor(x).long().to("cuda") for x in anomaly_detector_train_data[key]]
        else:
            anomaly_detector_train_data_dict[log_name] = [torch.Tensor(x).long().to("cuda") for x in anomaly_detector_train_data]
    
    for key in anomaly_detector_train_data_dict.keys():
        print(f"len(anomaly_detector_train_data_dict) (key={key}) = {len(anomaly_detector_train_data_dict[key])}")
    
    return anomaly_detector_train_data_dict


def load_train_labels_detector(anomaly_detector_train_data_dict, params):
    """
    Load train labels for the anomaly detector from specified paths.

    Parameters:
    - anomaly_detector_train_data_dict (dict): A dictionary containing train data for the anomaly detector.
    - params (dict): A dictionary containing configuration parameters, including label paths.

    Returns:
    dict: A dictionary where keys are log names, and values are torch tensors, nested dictionaries, or scalar values.

    Example:
    ```python
    detector_params = {
        'anomaly_detector': {
            'train_paths': ['/path/to/log_data_1.pkl', '/path/to/log_data_2.pkl'],
            'label_paths': ['/path/to/labels_1.pkl', '/path/to/labels_2.pkl'],
            ...
        },
        ...
    }
    detector_train_data = load_train_data_detector(anomaly_detector_train_data_dict, detector_params)
    detector_train_labels = load_train_labels_detector(detector_train_data, detector_params)
    ```

    Note:
    This function loads train labels for the anomaly detector from specified pickle files.
    It organizes the labels into a dictionary, where keys are log names derived from file names, and values are torch tensors, nested dictionaries, or scalar values.
    For log names containing "HDFS_1" or "hadoop" (case-insensitive), the function creates nested dictionaries to organize labels according to their respective subcategories.
    The loaded labels are either single torch tensors, nested dictionaries containing block-level labels, or scalar values for non-HDFS or non-Hadoop logs.
    """
    anomaly_detector_train_labels_dict = {}
    for i in range(len(params["anomaly_detector"]["label_paths"])):
        with open(params["anomaly_detector"]["label_paths"][i], "rb") as file:
            anomaly_detector_train_labels = pickle.load(file)
        log_name = ".".join(params["anomaly_detector"]["train_paths"][i].split("/")[-1].split(".")[:-1])
        if "HDFS_1" in log_name:
            anomaly_detector_train_labels_dict[log_name] = {}
            for line in anomaly_detector_train_labels:
                block = line.split(",")[0].strip()
                if line.split(",")[-1].strip() == "Normal":
                    label = 0
                else: 
                    label = 1
                anomaly_detector_train_labels_dict[log_name][block] = label
        elif "hadoop"in log_name.lower():
            anomaly_detector_train_labels_dict[log_name] = {}
            block_label = {}
            for line in anomaly_detector_train_labels:
                block = line.split(",")[0].strip()
                if line.split(",")[-1].strip() == "Normal":
                    block_label[block] = 0
                else: 
                    block_label[block] = 1
            for container in anomaly_detector_train_data_dict[log_name].keys():
                application_substing = "_".join(container.split("_")[:3])
                anomaly_detector_train_labels_dict[log_name][container] = block_label[application_substing]
        else:
            anomaly_detector_train_labels_dict[log_name] = torch.Tensor(anomaly_detector_train_labels)
    return anomaly_detector_train_labels_dict


def load_test_data_detector(params):
    """
    Load test data for the anomaly detector from specified paths.

    Parameters:
    - params (dict): A dictionary containing configuration parameters, including test data paths.

    Returns:
    dict: A dictionary where keys are log names, and values are lists of torch tensors.

    Example:
    ```python
    detector_params = {
        'anomaly_detector': {
            'test_data_paths': ['/path/to/test_data_1.pkl', '/path/to/test_data_2.pkl'],
            ...
        },
        ...
    }
    detector_test_data = load_test_data_detector(detector_params)
    ```

    Note:
    This function loads test data for the anomaly detector from specified pickle files.
    It organizes the test data into a dictionary, where keys are log names derived from file names, and values are lists of torch tensors.
    The loaded test data is stored in the 'anomaly_detector_test_data_dict' dictionary, which can be used for further processing.
    """
    anomaly_detector_test_data_dict = {}
    for i in range(len(params["anomaly_detector"]["test_data_paths"])):
        with open(params["anomaly_detector"]["test_data_paths"][i], "rb") as file:
            anomaly_detector_test_data = pickle.load(file)
        log_name = ".".join(params["anomaly_detector"]["test_data_paths"][i].split("/")[-1].split(".")[:-1])
        anomaly_detector_test_data_dict[log_name] = [torch.Tensor(x).long().to("cuda") for x in anomaly_detector_test_data]
    
    for key in anomaly_detector_test_data_dict.keys():
        print(f"len(anomaly_detector_test_data_dict) (key={key}) = {len(anomaly_detector_test_data_dict[key])}")

    return anomaly_detector_test_data_dict


def load_test_labels_detector(params):
    """
    Load test labels for the anomaly detector from specified paths.

    Parameters:
    - params (dict): A dictionary containing configuration parameters, including test label paths.

    Returns:
    dict: A dictionary where keys are log names, and values are torch tensors representing the test labels.

    Example:
    ```python
    detector_params = {
        'anomaly_detector': {
            'test_labels': ['/path/to/test_labels_1.pkl', '/path/to/test_labels_2.pkl'],
            ...
        },
        ...
    }
    detector_test_labels = load_test_labels_detector(detector_params)
    ```

    Note:
    This function loads test labels for the anomaly detector from specified pickle files.
    It organizes the test labels into a dictionary, where keys are log names derived from file names, and values are torch tensors.
    The loaded test labels are stored in the 'anomaly_detector_test_labels_dict' dictionary, which can be used for further processing.
    """
    anomaly_detector_test_labels_dict = {}
    for i in range(len(params["anomaly_detector"]["test_labels"])):
        with open(params["anomaly_detector"]["test_labels"][i], "rb") as file:
            anomaly_detector_test_labels = pickle.load(file)
        log_name = ".".join(params["anomaly_detector"]["test_data_paths"][i].split("/")[-1].split(".")[:-1])
        anomaly_detector_test_labels_dict[log_name] = torch.Tensor(anomaly_detector_test_labels)

    for key in anomaly_detector_test_labels_dict.keys():
        print(f"anomaly_detector_test_labels_dict[{key}].shape = {anomaly_detector_test_labels_dict[key].shape}")

    return anomaly_detector_test_labels_dict


def encode_data(transformer_encoder, data, batch_size, pad_token_id):
    """
    Encode data using a transformer encoder.

    Parameters:
    - transformer_encoder: An instance of a transformer encoder.
    - data: The data to be encoded.
    - batch_size (int): The batch size to use during encoding.
    - pad_token_id: The padding token ID used during encoding.

    Returns:
    dict: A dictionary where keys represent log names, and values are torch tensors representing the encoded data.

    Example:
    ```python
    encoder = TransformerEncoder(...)  # Assume this is an already initialized transformer encoder
    input_data = {'HDFS_1': [...], 'Spark': [...], ...}
    batch_size = 32
    pad_token_id = 0
    encoded_data = encode_data(encoder, input_data, batch_size, pad_token_id)
    ```

    Note:
    This function encodes input data using the specified transformer encoder.
    The data is organized into a dictionary, where keys are log names and values are torch tensors representing the encoded data.
    The function prints information about the encoding process, including the shape of encoded data or the number of encoded blocks for specific logs.
    """
    encoded = transformer_encoder.encode(
        data=data,
        batch_size = batch_size, 
        padding_value = pad_token_id
    )
    for key in encoded.keys():
        if "HDFS_1" in key or "hadoop" in key.lower():
            print(f"Encoded {key} blocks : {len(encoded[key])}")
        else:
            print(f"Encoded data's shape (key={key}): {encoded[key].shape}")
    return encoded


def main(conf_path):
    ###################
    ### LOAD PARAMS ###
    ###################

    with open(conf_path) as f:
        params = yaml.load(f, Loader=SafeLoader)
    if params["neptune_logging"]:
        run = initialize_neptune()
        run["params"] = dict_to_neptune_loggable(copy.deepcopy(params))
    else:
        run = None

    ####################
    ### ENCODER PART ###
    ####################

    transformer_encoder = TransformerEncoder(
        run=run,
        load_path=params["transformer_encoder"]["load_path"],
        mask_ignore_token_ids = params["transformer_encoder"], 
        mask_token_id = params["transformer_encoder"]["mask_token_id"], 
        pad_token_id = params["transformer_encoder"]["pad_token_id"], 
        mask_prob = params["transformer_encoder"]["mask_prob"], 
        replace_prob = params["transformer_encoder"]["replace_prob"], 
        num_tokens = params["transformer_encoder"]["num_tokens"], 
        max_seq_len = params["transformer_encoder"]["max_seq_len"], 
        attn_layers_dim = params["transformer_encoder"]["attn_layers"]["dim"], 
        attn_layers_depth = params["transformer_encoder"]["attn_layers"]["depth"], 
        attn_layers_heads = params["transformer_encoder"]["attn_layers"]["heads"],
    )

    # Encoder training
    handle_encoder_training(transformer_encoder=transformer_encoder, params=params)

    #####################
    ### DETECTOR PART ###
    #####################

    # Load data for detector
    anomaly_detector_train_data_dict = load_train_data_detector(params=params)

    # Load labels for detector
    anomaly_detector_train_labels_dict = load_train_labels_detector(anomaly_detector_train_data_dict=anomaly_detector_train_data_dict, params=params)

    # Openstack label generation
    generate_openstack_labels(anomaly_detector_train_data_dict=anomaly_detector_train_data_dict, anomaly_detector_train_labels_dict=anomaly_detector_train_labels_dict)


    for key in anomaly_detector_train_labels_dict.keys():
        if "HDFS_1" in key or "hadoop" in key.lower():
            print(f"anomaly_detector_train_labels_dict[{key}] -> number of blocks = {len(anomaly_detector_train_labels_dict[key].keys())}")
            #print(f"anomaly_detector_train_labels_dict[{key}] -> number of lines = {sum(anomaly_detector_train_labels_dict[key].values())}")
        else:
            print(f"anomaly_detector_train_labels_dict[{key}].shape = {anomaly_detector_train_labels_dict[key].shape}")


    ### Loading test data
    anomaly_detector_test_data_dict = load_test_data_detector(params=params)
    anomaly_detector_test_labels_dict = load_test_labels_detector(params=params)

    # Encode loaded data with encoder
    encoded_train = encode_data(
        transformer_encoder=transformer_encoder, 
        data=anomaly_detector_train_data_dict, 
        batch_size=params["transformer_encoder"]["batch_size"],
        pad_token_id=params["transformer_encoder"]["pad_token_id"]
    )
    

    anomaly_detector = AnomalyDetector(
        run=run,
        load_path=params["anomaly_detector"]["load_path"],
        embed_dim = params["anomaly_detector"]["embed_dim"], 
        ff_dim = params["anomaly_detector"]["ff_dim"], 
        max_len = params["anomaly_detector"]["max_len"], 
        num_heads = params["anomaly_detector"]["num_heads"], 
        dropout = params["anomaly_detector"]["dropout"],
    )

    anomaly_detector.train(
        X=encoded_train, 
        Y=anomaly_detector_train_labels_dict, 
        train_val_test_split = params["anomaly_detector"]["train_val_test_split"], 
        save_path = params["anomaly_detector"]["save_path"], 
        lr = params["anomaly_detector"]["lr"], 
        lr_decay_step_size = params["anomaly_detector"]["lr_decay_step_size"], 
        lr_decay_gamma = params["anomaly_detector"]["lr_decay_gamma"], 
        early_stop_tolerance = params["anomaly_detector"]["early_stop_tolerance"], 
        early_stop_min_delta = params["anomaly_detector"]["early_stop_min_delta"], 
        batch_size = params["anomaly_detector"]["batch_size"], 
        epochs = params["anomaly_detector"]["epochs"],
        balancing_ratio=params["anomaly_detector"]["balancing_ratio"]
    )

    encoded_test = encode_data(
        transformer_encoder=transformer_encoder, 
        data=anomaly_detector_test_data_dict, 
        batch_size=params["transformer_encoder"]["batch_size"],
        pad_token_id=params["transformer_encoder"]["pad_token_id"]
    )

    anomaly_detector.eval(
        batch_size=params["anomaly_detector"]["batch_size"], 
        X_test=encoded_test,
        Y_test=anomaly_detector_test_labels_dict
    )

    if run is not None:
        run.stop()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No cuda device found! Exiting...")
        raise Exception("NO CUDA ERROR")
    
    ####################
    ### PARSE PARAMS ###
    ####################

    parser = argparse.ArgumentParser(
                    prog='UnifiedLog',
                    description='This script trains an ecoder on multiple datasets, and then trains an anomaly detector.',
                    )

    parser.add_argument('-c', '--config')       # option that takes a value
    parser.add_argument('-t', '--threads')
    args = parser.parse_args()
    torch.set_num_threads(int(args.threads))

    if args.config.endswith(".yaml"):
        main(conf_path=args.config)
    else:
        for conf in os.listdir(args.config):
            main(conf_path=os.path.join(args.config, conf))
    
    