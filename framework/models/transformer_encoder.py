from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch
import os
import torch
import time
from torch.optim import AdamW
from mlm_pytorch import MLM
import pickle
from x_transformers import TransformerWrapper, Encoder

class BatchGenerator(Dataset):
    def __init__(self, X, batch_size, max_len, padding_value):
        self.X = X
        self.batch_size = batch_size
        self.max_len = max_len
        self.padding_value = padding_value

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        seqs = [tens.squeeze() for tens in self.X[idx * self.batch_size : min((idx + 1) * self.batch_size, len(self.X))]]
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=self.padding_value)
        return padded_seq_batch[:,:self.max_len].to("cuda")

class TransformerEncoder():
    def __init__(self, run, load_path, mask_ignore_token_ids, mask_token_id, pad_token_id, mask_prob, replace_prob, num_tokens, max_seq_len, attn_layers_dim, attn_layers_depth, attn_layers_heads):
        if load_path is None:
            self.transformer = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len,
                attn_layers = Encoder(
                    dim = attn_layers_dim,
                    depth = attn_layers_depth,
                    heads = attn_layers_heads
                )
            )
        else:
            print(f"Loading transformer encoder...({load_path})")
            self.transformer = pickle.load(open(load_path, "rb"))

        self.vocab_size = num_tokens
        self.max_len = max_seq_len
        self.attn_layers_dim = attn_layers_dim
        self.run = run

        self.trainer = MLM(
            self.transformer,
            mask_token_id = mask_token_id,          # the token id reserved for masking
            pad_token_id = pad_token_id,           # the token id for padding
            mask_prob = mask_prob,           # masking probability for masked language modeling
            replace_prob = replace_prob,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
            #mask_ignore_token_ids = [data[0][-1], data[0][0]]  # other tokens to exclude from masking, include the [cls] and [sep] here
            mask_ignore_token_ids = mask_ignore_token_ids  # other tokens to exclude from masking, include the [cls] and [sep] here
        ).to("cuda")#.cuda()


    def train(self, data, train_val_test_split, batch_size, lr, save_path, epochs, padding_value, save_every_epoch):
        """
        Trains the transformer encoder.

        Parameters:
        - data (dict): A dictionary containing training data, where keys are log source names, and values are lists of PyTorch tensors.
        - train_val_test_split (list): A list representing the ratio of data to use for training, validation, and test sets.
        - batch_size (int): The batch size for training.
        - lr (float): The learning rate for training.
        - save_path (str): The path to save the trained model.
        - epochs (int): The number of training epochs.
        - padding_value: The padding token ID used during training.
        - save_every_epoch (bool): If True, save the model at the end of each epoch.

        Returns:
        None

        Example:
        ```python
        transformer = TransformerEncoder(...)
        training_data = {'HDFS_1': [...], 'Spark': [...], ...}
        train_val_test_split = [0.8, 0.1, 0.1]
        batch_size = 32
        lr = 0.001
        save_path = "/path/to/save/models"
        epochs = 10
        padding_value = 0
        save_every_epoch = True
        transformer.train(training_data, train_val_test_split, batch_size, lr, save_path, epochs, padding_value, save_every_epoch)
        ```

        Note:
        This method trains the transformer encoder using the specified training data, splitting it into training and validation sets.
        It uses the AdamW optimizer, logs training and validation losses, and saves the model at the specified intervals.
        """
        print("Starting training of transformer encoder...")
        concatenated_data = []
        for key in data.keys():
            concatenated_data = concatenated_data + data[key][:int(len(data[key])*train_val_test_split[0])]
        train_generator = BatchGenerator(
            concatenated_data, 
            batch_size=batch_size, 
            max_len=self.max_len,
            padding_value=padding_value
        )
        val_generators = {}
        for key in data.keys():
            val_generator = BatchGenerator(
                data[key][int(len(data[key])*train_val_test_split[0]):int(len(data[key])*train_val_test_split[1])], 
                batch_size=batch_size, 
                max_len=self.max_len,
                padding_value=padding_value
            )
            val_generators[key] = val_generator

        train_loader = DataLoader(train_generator, num_workers=0, shuffle=True)
        val_loaders = {}
        for key in data.keys():
            val_loader = DataLoader(val_generators[key], num_workers=0, shuffle=True)
            val_loaders[key] = val_loader

        opt = AdamW(self.trainer.parameters(), lr=lr)
        all_train_losses = []
        all_val_losses = {}
        for key in data.keys():
            all_val_losses[key] = []

        t0 = time.time()
        step = 0
        for epoch in range(epochs):
            train_losses = []
            val_losses = {}
            for key in val_loaders.keys():
                val_losses[key] = []
            for _, x in enumerate(train_loader):
                step += 1
                loss = self.trainer(x[0])
                loss.backward()
                train_losses.append(loss.item())
                opt.step()
                opt.zero_grad()
                if (step*batch_size) % min([2500000, len(train_loader)*batch_size]) <= batch_size:
                    with torch.no_grad():
                        for key in val_loaders.keys():
                            for _, x in enumerate(val_loaders[key]):
                                loss = self.trainer(x[0])
                                val_losses[key].append(loss.item())
                        for key in val_loaders.keys():
                            all_val_losses[key].append(sum(val_losses[key])/len(val_losses[key]))       
                    all_train_losses.append(sum(train_losses)/len(train_losses))
                    print(f"Epoch: {epoch} step {step}:\n\tTrain loss: \t\t\t\t{all_train_losses[-1]:.04f}")
                    for key in val_loaders.keys():
                            print(f"\tVal loss on {key}: \t\t{all_val_losses[key][-1]:.04f}")
                    if self.run is not None:
                        self.run["transformer_encoder"]["train_loss"].log(all_train_losses[-1])
                        for key in val_loaders.keys():
                            self.run["transformer_encoder"]["val_loss"][key].log(all_val_losses[key][-1])
            if save_path is not None:
                _save_path = save_path
                if save_every_epoch:
                    _save_path = ".".join(_save_path.split(".")[:-1]) + f"_epoch_{epoch+1}." + _save_path.split(".")[-1]
                if not os.path.isdir("/".join(_save_path.split("/")[:-1])):
                    os.mkdir("/".join(_save_path.split("/")[:-1])+"/")
                self.run["transformer_encoder"]["save_path"] = _save_path
                pickle.dump(self.transformer, open(_save_path, "wb"))
        print(f"Training took {time.time()-t0:.2f}s")


    def encode(self, data, batch_size, padding_value):
        """
        Encodes data using the trained transformer encoder.

        Parameters:
        - data (dict): A dictionary containing log data, where keys are log source names, and values are either lists of PyTorch tensors or nested dictionaries.
        - batch_size (int): The batch size for encoding.
        - padding_value: The padding token ID used during encoding.

        Returns:
        dict: A dictionary where keys represent log names, and values are torch tensors representing the encoded data.

        Example:
        ```python
        transformer = TransformerEncoder(...)
        input_data = {'HDFS_1': {...}, 'Spark': {...}, ...}
        batch_size = 32
        padding_value = 0
        encoded_data = transformer.encode(input_data, batch_size, padding_value)
        ```

        Note:
        This method encodes input data using the trained transformer encoder.
        The data is organized into a dictionary, where keys are log names, and values are torch tensors representing the encoded data.
        The function prints information about the encoding process, including the shape of the encoded data or the number of encoded blocks for specific logs.
        """
        print("Encoding data for anomaly detector...")
        loaders = {}
        for key in data.keys():
            if "HDFS_1" in key or "hadoop" in key.lower():
                loaders[key] = {}
                for block in data[key].keys():
                    generator = BatchGenerator(
                    data[key][block], 
                    batch_size=batch_size, 
                    max_len=self.max_len,
                    padding_value=padding_value
                    )
                    loader = DataLoader(generator, num_workers=0, shuffle=False)
                    loaders[key][block] = loader
            else:
                generator = BatchGenerator(
                    data[key], 
                    batch_size=batch_size, 
                    max_len=self.max_len,
                    padding_value=padding_value
                )
                loader = DataLoader(generator, num_workers=0, shuffle=False)
                loaders[key] = loader

        all_encoded = {}
        for key in data.keys():
            if "HDFS_1" in key or "hadoop" in key.lower():
                all_encoded[key] = {}
                for block in data[key].keys():
                    all_encoded[key][block] = torch.empty((1,self.attn_layers_dim))
            else:
                all_encoded[key] = torch.empty((1,self.attn_layers_dim))

        self.transformer = self.transformer.eval()
        with torch.no_grad():
            for key in data.keys():
                if "HDFS_1" in key or "hadoop" in key.lower():
                    for block in loaders[key]:
                        for _, log in enumerate(loaders[key][block]):
                            
                            mask = torch.ones_like(log[0]).bool()
                            encoded = self.transformer(log[0].cuda(), mask=mask.cuda(), return_embeddings = True)[:,0,:]
                            all_encoded[key][block] = torch.cat((all_encoded[key][block], encoded.cpu()), dim=0)
                        all_encoded[key][block] = all_encoded[key][block][1:,:]
                else:
                    for _, log in enumerate(loaders[key]):
                        mask = torch.ones_like(log[0]).bool()
                        encoded = self.transformer(log[0].cuda(), mask=mask.cuda(), return_embeddings = True)[:,0,:]
                        all_encoded[key] = torch.cat((all_encoded[key], encoded.cpu()), dim=0)
                    all_encoded[key] = all_encoded[key][1:,:]

        for key in data.keys():
            if "HDFS_1" in key or "hadoop" in key.lower():
                print(f"Encoded {key} blocks={len(data[key])}")
            else:
                print(f"Encoded {key} lines: {all_encoded[key].shape[0]}")
        return all_encoded
    