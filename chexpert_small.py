#!/usr/bin/env python
"""
chexpert_small.py

Defines the ChexpertSmall dataset class for the CheXpert-v1.0-small dataset.
This module handles downloading, extracting, and processing the dataset from CSV files,
and provides utility functions for extracting patient IDs and computing dataset statistics.

Attributes:
    url (str): URL to download the CheXpert-v1.0-small dataset.
    dir_name (str): Name of the directory created after extraction.
    attr_all_names (list): All available attribute names.
    attr_names (list): Selected competition labels used for training and evaluation.

Functions:
    extract_patient_ids(dataset, idxs): Extracts patient IDs from image paths.
    compute_mean_and_std(dataset): Computes the mean and standard deviation of images in the dataset.

Usage:
    Run this module as a script to perform a simple test:
      - Load the training dataset.
      - Optionally display a few images from the validation set along with their labels.
      - Compute and print dataset statistics.
"""

import os
import sys
from urllib import request
import zipfile
import json
import math

import pandas as pd
from pandas.core.internals.managers import BlockManager
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChexpertSmall(Dataset):
    """
    ChexpertSmall dataset class for the CheXpert-v1.0-small dataset.

    This class downloads and extracts the dataset if necessary, processes the CSV files
    to create training and validation sets (as .pt files), and supports different modes:
      - 'train', 'valid': For training and evaluation.
      - 'test': Loads a CSV file and returns dummy labels.
      - 'vis': Selects a subset for visualization.
    """
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # Directory name derived from the URL
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # Selected competition labels.
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None):
        """
        Initializes the ChexpertSmall dataset.

        Parameters:
            root (str): Root directory for the dataset.
            mode (str): One of 'train', 'valid', 'test', or 'vis' (for visualization).
            transform (callable, optional): Transformations to apply to the images.
            data_filter (dict, optional): Dictionary to filter data by attributes.
            mini_data (int, optional): If provided, limits the dataset size (useful for debugging).
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        assert mode in ['train', 'valid', 'test', 'vis']
        self.mode = mode

        if mode == 'test':
            # In test mode, the root is a CSV file and we return dummy labels.
            self.data = pd.read_csv(self.root, keep_default_na=True)
            self.root = '.'  # Base path for images.
            self.data[self.attr_names] = pd.DataFrame(np.zeros((len(self.data), len(self.attr_names))))
        else:
            self._maybe_download_and_extract()
            self._maybe_process(data_filter)

            data_file = os.path.join(self.root, self.dir_name, 
                                     'valid.pt' if mode in ['valid', 'vis'] else 'train.pt')
            with torch.serialization.safe_globals([pd.DataFrame, BlockManager]):
                self.data = torch.load(data_file, weights_only=False)

            if mini_data is not None:
                # Use a subset of the data for debugging.
                self.data = self.data[:mini_data]

            if mode == 'vis':
                # For visualization: select a few examples for each condition category.
                idxs = []
                for attr in self.attr_names:
                    # Single condition examples.
                    idxs.append(self.data.loc[(self.data[attr]==1) & (self.data[self.attr_names].sum(1)==1), self.attr_names].head(3).index.tolist())
                # No findings.
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==0, self.attr_names].head(3).index.tolist())
                # Two conditions.
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)==2, self.attr_names].head(3).index.tolist())
                # More than two conditions.
                idxs.append(self.data.loc[self.data[self.attr_names].sum(1)>2, self.attr_names].head(3).index.tolist())
                self.vis_attrs = self.attr_names + ['No findings', '2 conditions', 'Multiple conditions']
                self.vis_idxs = idxs
                # Flatten the list of indices.
                idxs_flatten = torch.tensor([i for sublist in idxs for i in sublist])
                self.data = self.data.iloc[idxs_flatten]

        # Store indices of selected attributes for faster access.
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding labels at the given index.

        Parameters:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, attribute labels, original index)
                - image (Tensor): Transformed image.
                - attribute labels (Tensor): Float tensor of labels.
                - original index: Index from the original dataframe.
        """
        # Load the image.
        img_path = self.data.iloc[idx, 0]  # First column contains image path.
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)
        # Retrieve attribute labels.
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)
        # Get the original index from the dataframe.
        orig_idx = self.data.index[idx]
        return img, attr, orig_idx

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.data)

    def _maybe_download_and_extract(self):
        """
        Downloads and extracts the dataset if not already present.
        """
        fpath = os.path.join(self.root, os.path.basename(self.url))
        if not os.path.exists(os.path.join(self.root, self.dir_name)):
            if not os.path.exists(fpath):
                print('Downloading ' + self.url + ' to ' + fpath)
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                request.urlretrieve(self.url, fpath, _progress)
                print()
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
                macosx_path = os.path.join(self.root, self.dir_name, '__MACOSX')
                if os.path.exists(macosx_path):
                    os.rmdir(macosx_path)
            os.unlink(fpath)
            print('Dataset extracted.')

    def _maybe_process(self, data_filter):
        """
        Processes CSV files to create .pt files for faster dataset loading.
        Processing includes:
            1. Filling missing labels (NaN) with 0.
            2. Replacing uncertain labels (-1) with 1 (using the U-Ones method).
            3. Applying optional attribute filters.
        """
        train_file = os.path.join(self.root, self.dir_name, 'train.pt')
        valid_file = os.path.join(self.root, self.dir_name, 'valid.pt')
        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'), keep_default_na=True)
            valid_df[self.attr_names] = valid_df[self.attr_names].fillna(0)
            valid_df[self.attr_names] = valid_df[self.attr_names].replace(-1, 1)

            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, self.dir_name, 'train.csv'), data_filter)
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        """
        Loads and preprocesses training data from a CSV file.

        Parameters:
            csv_path (str): Path to the training CSV file.
            data_filter (dict, optional): Dictionary specifying filters (e.g., {'Frontal/Lateral': 'Frontal'}).

        Returns:
            DataFrame: Preprocessed training data.
        """
        train_df = pd.read_csv(csv_path, keep_default_na=True)
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1, 1)
        if data_filter is not None:
            for k, v in data_filter.items():
                train_df = train_df[train_df[k] == v]
            with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
                json.dump(data_filter, f)
        return train_df


def extract_patient_ids(dataset, idxs):
    """
    Extracts patient IDs from the image paths in the dataset.

    Parameters:
        dataset (ChexpertSmall): An instance of the ChexpertSmall dataset.
        idxs (iterable): Indices for which to extract patient IDs.

    Returns:
        ndarray: Array of patient IDs (derived from the image paths).
    """
    return dataset.data['Path'].loc[idxs].str.rsplit('/', expand=True, n=1)[0].values


def compute_mean_and_std(dataset):
    """
    Computes the mean and standard deviation of images in the dataset.

    Parameters:
        dataset (Dataset): A dataset instance yielding image tensors.

    Returns:
        tuple: (mean, std) of the dataset.
    """
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset, desc="Computing mean and std"):
        x = img.mean().item()
        new_m = m + (x - m) / k
        s += (x - m) * (x - new_m)
        m = new_m
        k += 1
    print('Number of datapoints:', k)
    return m, math.sqrt(s / (k - 1))


def main():
    """
    Main function for testing the ChexpertSmall dataset.

    Loads the training dataset and prints its length.
    Optionally, outputs a few images from the validation set along with their labels,
    and computes dataset statistics.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data", help='Data directory.')
    args = parser.parse_args()

    ds = ChexpertSmall(root=args.data_dir, mode='train')
    print('Train dataset loaded. Length:', len(ds))

    output_dir = 'results/test/'

    # Display a few images from the validation set and print their labels.
    if True:
        import torchvision.transforms as T
        from torchvision.utils import save_image
        ds_valid = ChexpertSmall(root=args.data_dir, mode='valid',
                transform=T.Compose([T.CenterCrop(320), T.ToTensor(), T.Normalize(mean=[0.5330], std=[0.0349])]))
        print('Valid dataset loaded. Length:', len(ds_valid))
        for i in range(10):
            img, attr, patient_id = ds_valid[i]
            save_image(img, f'test_valid_dataset_image_{i}.png', normalize=True, scale_each=True)
            print('Patient id:', patient_id, '; labels:', attr)

    # Optionally compute mean and std of the training dataset.
    if False:
        ds_train = ChexpertSmall(root=args.data_dir, mode='train', transform=T.Compose([T.CenterCrop(320), T.ToTensor()]))
        m, s = compute_mean_and_std(ds_train)
        print('Dataset mean:', m, '; dataset std:', s)


if __name__ == '__main__':
    main()
