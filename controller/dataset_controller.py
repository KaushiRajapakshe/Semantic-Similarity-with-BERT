"""
    Created by KaushiRajapakshe on 15/08/2021.

    Data set Controller
"""
# Importing all required libraries to work with the dataset and model
import tensorflow as tf


class DatasetController:

    # Shape of the data
    def data_shape(self):
        print(f"Total train samples : {self.get_train_df().shape[0]}")
        print(f"Total validation samples: {self.get_valid_df().shape[0]}")
        print(f"Total test samples: {self.test_df.shape[0]}")

    # Display one sample from the dataset
    def display_sample_dataset(self):
        print(f"Sentence1: {self.get_train_df().loc[1, 'sentence1']}")
        print(f"Sentence2: {self.get_train_df().loc[1, 'sentence2']}")
        print(f"Similarity: {self.get_train_df().loc[1, 'similarity']}")

    # Preprocessing
    def data_preprocessing(self):
        # Check for existing null values
        print("Number of missing values")
        print(self.train_df.isnull().sum())
        # Dropping Nan entries in the train data
        self.train_df.dropna(axis=0, inplace=True)

    # Distribution of the training targets.
    def dis_train_target(self):
        print("Train Target Distribution", self.get_train_df().similarity.value_counts())

    # Distribution of our validation targets.
    def dis_validation_target(self):
        print("Validation Target Distribution", self.get_valid_df().similarity.value_counts())

    # Skipping the value "-" appears as part of the training and validation targets.
    def skip_hyphen_values(self):
        self.set_train_df(
            self.get_train_df()[self.get_train_df().similarity != "-"]
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
        )
        self.set_valid_df(
            self.get_valid_df()[self.get_valid_df().similarity != "-"]
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
        )

    def encode(self):
        # One-hot encode training, validation, and test labels.
        self.train_df["label"] = self.train_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.set_y_train(tf.keras.utils.to_categorical(self.train_df.label, num_classes=3))

        self.valid_df["label"] = self.valid_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.set_y_val(tf.keras.utils.to_categorical(self.valid_df.label, num_classes=3))

        self.test_df["label"] = self.test_df["similarity"].apply(
            lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
        )
        self.set_y_test(tf.keras.utils.to_categorical(self.test_df.label, num_classes=3))
