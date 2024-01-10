import argparse
import os
import pandas as pd
import string
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle
import numpy as np


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


def replace_non_ascii(x):
    return ''.join('[NONA]' if char not in string.printable else char for char in x)


def replace_num(x):
    for i in range(10):
        x = x.replace(str(i), "[NUM]")
    return x


def read_file_contents(file, max_log_lines, save_path, verbose=True):
    """
    Read the contents of a log file, and optionally save labels.

    Parameters:
    - file (str): The path to the log file to be read.
    - max_log_lines (int): The maximum number of log lines to read from the file.
    - save_path (str): The directory where labels will be saved (if applicable).
    - verbose (bool): If True, print loading information.

    Returns:
    list: A list containing log lines from the file.

    Example:
    ```python
    file = "/path/to/log_file.log"
    max_log_lines = 1000
    save_path = "/path/to/save/folder"
    verbose = True
    log_lines = read_file_contents(file, max_log_lines, save_path, verbose)
    ```

    Note:
    The function reads the contents of a log file, tracks labels (if applicable), and returns a list containing log lines.
    Labels are tracked for specific log files (e.g., "Thunderbird" or "BGL"), and if labels are tracked, they are saved to a 'labels' subdirectory within the specified save folder.
    The function attempts to load the file with both "utf-8" and "cp1252" encodings, choosing the first successful attempt.
    """
    def read_lines_from_file(file, data, encoding):
        with open(file, encoding=encoding) as f:
            labels = []
            if "Thunderbird" in file or "BGL" in file:
                track_labels = True
            else:
                track_labels = False
            idx = 0
            for line in f:
                idx += 1
                if line[0] == "-":
                    line = line[1:]
                    if track_labels:
                        labels.append(1)
                elif track_labels and len(line.strip()) > 0:
                    labels.append(0)
                if len(line.strip()) > 0:
                    data.append(line.strip())
                if idx == max_log_lines:
                    break
            if track_labels:
                if not os.path.exists(os.path.join(save_path, "labels")):
                    os.makedirs(os.path.join(save_path, "labels"))
                with open(os.path.join(save_path, f"labels/{file.split('/')[-1].split('.')[0]}_labels.pkl"), "wb") as save_file:
                    pickle.dump(labels, save_file)
    
    try:
        data = [] 
        if verbose:
            print(f"Loading (encoding=utf-8) {file}")
        read_lines_from_file(file=file, data=data, encoding="utf-8")
    except:
        data = [] 
        if verbose:
            print(f"Loading (encoding=cp1252) {file}")
        read_lines_from_file(file=file, data=data, encoding="cp1252")
    if verbose:
        print(f"Loaded {len(data)} number of lines.")
    return data


def train_tokenizer(vocab_size, save_folder, data_df, ascii_policy, num_policy):
    """
    Train a tokenizer using the WordPiece model on a DataFrame containing log texts.

    Parameters:
    - vocab_size (int): The size of the vocabulary for the tokenizer.
    - save_folder (str): The directory where the trained tokenizer and intermediate files will be saved.
    - data_df (pd.DataFrame): A DataFrame containing log texts in a column named 'log_text'.

    Returns:
    - Tokenizer: The trained tokenizer.

    Example:
    ```python
    vocab_size = 10000
    save_folder = "/path/to/save/folder"
    data_df = pd.read_csv("/path/to/data.csv")
    trained_tokenizer = train_tokenizer(vocab_size, save_folder, data_df)
    ```

    Note:
    The function assumes that the input DataFrame contains a column named 'log_text' with log texts.
    """
    data_df[:int(len(data_df)*0.8)].to_csv(os.path.join(save_folder, "./sample_logs_train.txt"), escapechar='\\', header=False, index=False)
    data_df[int(len(data_df)*0.8):int(len(data_df)*0.9)].to_csv(os.path.join(save_folder, "./sample_logs_val.txt"), escapechar='\\', header=False, index=False)
    data_df[int(len(data_df)*0.9):].to_csv(os.path.join(save_folder, "./sample_logs_test.txt"), header=False, escapechar='\\', index=False)

    files = [os.path.join(save_folder, f"sample_logs_{split}.txt") for split in ["test", "train", "val"]]

    print(f"========Using vocabulary from {files}=======")

    unk_token = "[UNK]"  # token for unknown words
    spl_tokens = ["[CLS]","[SEP]", "[UNK]"]

    if num_policy == "num_special_char":
        spl_tokens.append("[NUM]")
    elif num_policy == "0_9_special_char":
        for i in range(10):
            spl_tokens.append(str(i))
            
    if ascii_policy == "special_char":
        spl_tokens.append("[NONA]")

    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, 
        special_tokens=spl_tokens,
        show_progress=True
    )
    tokenizer.pre_tokenizer = Whitespace()
    
    tokenizer.train(files, trainer) # training the tokenzier
    tokenizer.save(os.path.join(save_folder, "./tokenizer-trained.json"))
    tokenizer = Tokenizer.from_file(os.path.join(save_folder, "./tokenizer-trained.json"))
    return tokenizer


def get_data_to_tokenize(data_folder, max_log_lines, save_path):
    """
    Recursively reads log files from a specified data folder, extracts their contents, and organizes them by data source.

    Parameters:
    - data_folder (str): The top-level directory containing log files and subdirectories.
    - max_log_lines (int): The maximum number of lines to read from each log file.
    - save_path (str): The directory where intermediate results will be saved.

    Returns:
    dict: A dictionary where keys are log sources and values are lists of log lines.

    Example:
    ```python
    data_folder = "/path/to/data"
    max_log_lines = 1000
    save_path = "/path/to/save/folder"
    data_to_tokenize = get_data_to_tokenize(data_folder, max_log_lines, save_path)
    ```

    Note:
    The function organizes log data by log source. It reads log files from Hadoop, Spark, HDFS_v2 directories and extracts log lines from .log files inside those subdirectories.
    For other directories, it assumes the presence of ".log" files, extracting log lines and organizing them by file name.
    The total number of lines to tokenize is printed for reference.
    """
    data_to_tokenize = {}
    for file in os.listdir(data_folder):
        inner_file = os.path.join(data_folder, file)
        if os.path.isdir(inner_file):
            if inner_file.endswith("Hadoop") or inner_file.endswith("Spark") or inner_file.endswith("HDFS_v2"):
                print(f"Loading {inner_file}")
                data_to_tokenize[file] = []
                for inner_inner_file in os.listdir(inner_file):
                    if os.path.isdir(os.path.join(inner_file, inner_inner_file)):
                        print(f"\t/{inner_inner_file}")
                        inner_inner_folder = os.path.join(inner_file, inner_inner_file)
                        for logfile in os.listdir(inner_inner_folder):
                            data_to_tokenize[file] = data_to_tokenize[file] + read_file_contents(os.path.join(inner_inner_folder, logfile), max_log_lines=max_log_lines, verbose=False, save_path=save_path)
                    if len(data_to_tokenize[file]) > max_log_lines:
                        data_to_tokenize[file] = data_to_tokenize[file][:max_log_lines]
                        break
            else:
                for inner_inner_file in os.listdir(inner_file):
                    if os.path.join(inner_file, inner_inner_file).endswith(".log"):
                        data_to_tokenize[inner_inner_file.split(".")[0]] = read_file_contents(os.path.join(inner_file, inner_inner_file), max_log_lines=max_log_lines, save_path=save_path)
    print(f"Number of lines of logs to tokenize: {sum(len(x) for x in data_to_tokenize.values())}.")
    return data_to_tokenize


def tokenize_files(files_dict, save_folder, trained_tokenizer):
    """
    Tokenize log files using a trained tokenizer and save the tokenized sequences to pickle files.

    Parameters:
    - files_dict (dict): A dictionary where keys are log sources and values are lists of log lines.
    - save_folder (str): The directory where the tokenized files will be saved.
    - trained_tokenizer (Tokenizer): The trained tokenizer to be used for tokenization.

    Returns:
    None

    Example:
    ```python
    files_dict = {
        'Hadoop': ['log line 1', 'log line 2', ...],
        'Spark': ['log line 1', 'log line 2', ...],
        ...
    }
    save_folder = "/path/to/save/folder"
    trained_tokenizer = Tokenizer()  # Assume this is an already trained tokenizer
    tokenize_files(files_dict, save_folder, trained_tokenizer)
    ```

    Note:
    The function tokenizes log lines using a trained tokenizer. It adds special tokens '[CLS]' and '[SEP]' to each line,
    replaces numbers with a special token '[NUM]', removes non-ASCII characters, and saves the resulting tokenized sequences to pickle files.
    Tokenized files are saved in a subdirectory named 'tokenized' within the specified save folder.
    """
    if not os.path.exists(os.path.join(save_folder, f"tokenized")):
        os.makedirs(os.path.join(save_folder, f"tokenized"))
    for key in files_dict.keys():
        print(f"Tokenizing {key}")
        encoded = []
        for line in files_dict[key]:
            if line[0] == "-":
                line = line[1:]
            if len(line.strip()) > 0:
                encoded.append(trained_tokenizer.encode("[CLS]"+replace_num(remove_non_ascii(line))+"[SEP]").ids)
        with open(os.path.join(save_folder, f"tokenized/{key}_NUM_special_token.pkl"), "wb") as save_file:
            pickle.dump(encoded, save_file)


def tokenize_HDFS_for_detector(data_path, tokenizer, save_path, max_log_lines):
    """
    Tokenize HDFS log data with block structure for a detector.

    Parameters:
    - data_path (str): The path to the directory containing HDFS log data and labels.
    - tokenizer (Tokenizer): The trained tokenizer to be used for tokenization.
    - save_path (str): The directory where the tokenized data will be saved.
    - max_log_lines (int): The maximum number of log lines to process.

    Returns:
    None

    Example:
    ```python
    data_path = "/path/to/HDFS_data"
    tokenizer = Tokenizer()  # Assume this is an already trained tokenizer
    save_path = "/path/to/save/folder"
    max_log_lines = 10000
    tokenize_HDFS_for_detector(data_path, tokenizer, save_path, max_log_lines)
    ```

    Note:
    The function tokenizes HDFS log data with a block structure for the detector. It extracts block names and labels from an 'anomaly_label.csv' file.
    It then tokenizes log lines from the 'HDFS.log' file, organizing them by block name. Tokenized sequences are saved to pickle files, and labels are saved separately.
    Blocks with less than or equal to 20 log lines are discarded, and the resulting tokenized sequences are saved in a subdirectory named 'tokenized' within the specified save folder.
    """
    print("Tokenizing HDFS with the block structure for the detector.")
    good_blocks = []
    block_names = []
    labels = []

    with open(os.path.join(data_path, "HDFS_v1/preprocessed/anomaly_label.csv")) as file:
        for line in file:
            if line.split(",")[-1].strip() == "Normal":
                good_blocks.append(line.split(",")[0].strip())
            block_names.append(line.split(",")[0].strip())
            labels.append(line.strip())

    if not os.path.exists(os.path.join(save_path, "labels")):
        os.makedirs(os.path.join(save_path, "labels"))
    with open(os.path.join(save_path, "labels/anomaly_label_HDFS_1.pkl"), "wb") as save_file:
        pickle.dump(labels[1:], save_file)
    blocks = {}

    for b in block_names:
        blocks[b] = []

    with open(os.path.join(data_path, "HDFS_v1/HDFS.log")) as file:
        idx = 0
        for line in file:
            if idx == max_log_lines:
                break
            idx += 1
            if idx % 25000 == 0:
                print(f"At index={idx}", end="\r")
            for word in line.split(" "):
                if word.startswith("blk_"):
                    blocks[word.strip().replace(".", "")].append(line)
    
    less_than_20 = []
    encoded_dict = {}

    for key in blocks.keys():
        encoded = []
        for line in blocks[key]:
            line = line.strip()
            line = remove_non_ascii(line)
            line = replace_num(line)
            line = "[CLS]" + line + "[SEP]"
            encoded.append(tokenizer.encode(line).ids)
        if len(blocks[key]) > 20:
            encoded_dict[key] = encoded
        else:
            less_than_20.append(key)
    
    print(f"Throwing away {len(less_than_20)} blocks (length <= 20).")
    with open(os.path.join(save_path, "tokenized/HDFS_1_block_dict_NUM_special_token.pkl"), "wb") as save_file:
        pickle.dump(encoded_dict, save_file)


def tokenize_Hadoop_for_detector(data_path, tokenizer, save_path, max_log_lines):
    """
    Tokenize Hadoop log data with block structure for a detector.

    Parameters:
    - data_path (str): The path to the directory containing Hadoop log data and labels.
    - tokenizer (Tokenizer): The trained tokenizer to be used for tokenization.
    - save_path (str): The directory where the tokenized data will be saved.
    - max_log_lines (int): The maximum number of log lines to process.

    Returns:
    None

    Example:
    ```python
    data_path = "/path/to/Hadoop_data"
    tokenizer = Tokenizer()  # Assume this is an already trained tokenizer
    save_path = "/path/to/save/folder"
    max_log_lines = 10000
    tokenize_Hadoop_for_detector(data_path, tokenizer, save_path, max_log_lines)
    ```

    Note:
    The function tokenizes Hadoop log data with a block structure for a detector. It extracts labels from 'abnormal_label.txt'.
    It then tokenizes log lines from individual Hadoop log files, organizing them by application ID. Tokenized sequences are saved to pickle files, and labels are saved separately.
    Blocks with less than or equal to 20 log lines are discarded, and the resulting tokenized sequences are saved in a subdirectory named 'tokenized' within the specified save folder.
    """
    print("Tokenizing Hadoop with the block structure for the detector.")
    labels = []
    in_normal = False
    with open(os.path.join(data_path, "Hadoop/abnormal_label.txt")) as file:
        for line in file:
            if line.strip() == "Normal:":
                in_normal = True
            elif in_normal:
                if "application_" in line:
                    labels.append(f"{line.strip().split(' ')[1]},Normal")
                else:
                    in_normal = False
            elif not in_normal and "application_" in line:
                labels.append(f"{line.strip().split(' ')[1]},Abnormal")
    print(f"Found {len(labels)} number of applications in Hadoop.")

    if not os.path.exists(os.path.join(save_path, "labels")):
        os.makedirs(os.path.join(save_path, "labels"))
    with open(os.path.join(save_path, "labels/abnormal_label_hadoop.pkl"), "wb") as save_file:
        pickle.dump(labels, save_file)

    hadoop_apps = {}

    number_of_lines = 0
    hadoop_path = os.path.join(data_path, "Hadoop")
    for file in os.listdir(hadoop_path):
        if "application" in file:
            for innerfile in os.listdir(os.path.join(hadoop_path,file)):
                hadoop_apps[file + "_" + innerfile.split(".")[0]] = []
                with open(os.path.join(os.path.join(hadoop_path,file),innerfile)) as logfile:
                    for line in logfile:
                        hadoop_apps[file + "_" + innerfile.split(".")[0]].append(line.strip())
                        number_of_lines += 1
        if number_of_lines >= max_log_lines:
            break

    encoded_dict = {}
    less_than_20 = 0

    for key in hadoop_apps.keys():
        encoded = []
        for line in hadoop_apps[key]:
            #print(line)
            line = line.strip()
            line = remove_non_ascii(line)
            line = replace_num(line)
            line = "[CLS]" + line + "[SEP]"
            encoded.append(tokenizer.encode(line).ids)
        #pickle.dump(encoded, open(os.path.join(SAVE_PATH, key+".pkl"), "wb"))
        if len(encoded) > 20:
            encoded_dict[key] = encoded
        else:
            less_than_20 += 1
    print(F"Threw away {less_than_20} blocks with messages <= 20.")
    if not os.path.exists(os.path.join(save_path, f"tokenized")):
        os.makedirs(os.path.join(save_path, f"tokenized"))
    with open(os.path.join(save_path, "tokenized/Hadoop_for_detector_block_dict_NUM_special_token.pkl"), "wb") as save_file:
        pickle.dump(encoded_dict, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='data_preprocess',
        description='This script preprocesses the datasets downloaded from Loghub. \n ascii_policy: remove/special_char \n num_policy: special_char/keep',
    )
    parser.add_argument('-d', '--data_folder', required=True)
    parser.add_argument('-s', '--save_folder', required=True) 
    parser.add_argument('-v', '--vocab_size', default=1002) 
    parser.add_argument('-l', '--max_log_lines', default=5000000, type=int)
    parser.add_argument('-a', '--ascii_policy', required=True, choices=["remove", "special_char"])
    parser.add_argument('-n', '--num_policy', required=True, choices=["0_9_special_char", "num_special_char"])
    args = parser.parse_args()

    # Create the save folder if it doesn't exist
    if not os.path.exists(args.save_folder):
        print(f"Creating {args.save_folder} folder as it doesn't exist...")
        os.makedirs(args.save_folder)

    data_to_tokenize = get_data_to_tokenize(data_folder=args.data_folder, max_log_lines=args.max_log_lines, save_path=args.save_folder)

    # Create dataframe and suffle rows to mix data for the tokeniaztion
    data_df = pd.DataFrame({'log_text': [item for sublist in data_to_tokenize.values() for item in sublist]})

    if args.ascii_policy == "remove":
        print("Removing non-ascii characters from logs...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : remove_non_ascii(x))
    elif args.ascii_policy == "special_char":
        print("Replacing non-ascii characters with special token [NONA]...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : replace_non_ascii(x))
    else:
        raise Exception("Wrong ascii policy set in arg")

    if args.num_policy == "num_special_char":
        print("Replacing numbers with the [NUM] special character in the logs...")
        data_df['log_text'] = data_df['log_text'].apply(lambda x : replace_num(x))
    elif args.num_policy == "0_9_special_char":
        print("Keeping numbers as special characters in logs...")
    else:
        raise Exception("Wrong num policy set in arg")

    trained_tokenizer = train_tokenizer(vocab_size=args.vocab_size, save_folder=args.save_folder, data_df=data_df, ascii_policy=args.ascii_policy, num_policy=args.num_policy)

    tokenize_files(files_dict=data_to_tokenize, save_folder=args.save_folder, trained_tokenizer=trained_tokenizer)

    tokenize_HDFS_for_detector(data_path=args.data_folder, tokenizer=trained_tokenizer, save_path=args.save_folder, max_log_lines=args.max_log_lines)

    tokenize_Hadoop_for_detector(data_path=args.data_folder, tokenizer=trained_tokenizer, save_path=args.save_folder, max_log_lines=args.max_log_lines)
