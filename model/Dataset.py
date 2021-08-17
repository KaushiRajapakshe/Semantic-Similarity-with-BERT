"""
    Created by KaushiRajapakshe on 15/08/2021.

    Dataset
        - Download dataset from https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus
        - Save the downloaded csv file inside the Dataset Folder

        Dataset Overview:
        - sentence1: A man and a woman are sitting at a table outside, next to a small flower garden.
        - sentence2: The woman is having lunch with another woman.
        - similarity: This is the label chosen by the majority of annotators.

        Here are the "similarity" label values in the dataset:
        - Contradiction: The sentences share no similarity.
        - Entailment: The sentences have similar meaning.
        - Neutral: The sentences are neutral.

"""

# Importing all required libraries to work with the dataset and model
import pandas as pd
from constant.constant import dataset_train, dataset_dev, dataset_test


# There are more than 550k samples in total; we will use 100k for this example.
class Dataset:

    def __init__(self):
        self.train_df = pd.read_csv(dataset_train, nrows=1000)
        self.valid_df = pd.read_csv(dataset_dev, nrows=1000)
        self.test_df = pd.read_csv(dataset_test, nrows=1000)
        self.y_train = ''
        self.y_val = ''
        self.y_test = ''
        self.train_data = ''
        self.valid_data = ''

    # getter method train data set
    def get_train_df(self):
        return self.train_df

    # setter method train data set
    def set_train_df(self, train_df):
        self.train_df = train_df

    # getter method valid data set
    def get_valid_df(self):
        return self.valid_df

    # setter method valid data set
    def set_valid_df(self, valid_df):
        self.valid_df = valid_df

    # getter method test data set
    def get_test_df(self):
        return self.test_df

    # setter method test data set
    def set_test_df(self, test_df):
        self.test_df = test_df

    # getter method y train data set
    def get_y_train(self):
        return self.y_train

    # setter method y train data set
    def set_y_train(self, y_train):
        self.y_train = y_train

    # getter method y valid data set
    def get_y_val(self):
        return self.y_val

    # setter method y valid data set
    def set_y_val(self, y_val):
        self.y_val = y_val

    # getter method y test data set
    def get_y_test(self):
        return self.y_test

    # setter method y test data set
    def set_y_test(self, y_test):
        self.y_test = y_test

    # getter method train data set
    def get_train_data(self):
        return self.train_data

    # setter method train data set
    def set_train_data(self, train_data):
        self.train_data = train_data

# getter method valid data set
    def get_valid_data(self):
        return self.valid_data

    # setter method valid data set
    def set_valid_data(self, valid_data):
        self.valid_data = valid_data
