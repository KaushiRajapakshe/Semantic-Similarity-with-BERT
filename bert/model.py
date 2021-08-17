"""
    Created by KaushiRajapakshe on 15/08/2021.

    Semantic Similarity with BERT
    Natural Language Inference by fine-tuning BERT model on SNLI (Stanford Natural Language Inference) Corpus.

    * BERT model that takes two sentences as inputs and that outputs a similarity score for these two sentences.

"""

# Importing all required libraries to work with the dataset and model
import numpy as np
import pandas as pd
import tensorflow as tf


