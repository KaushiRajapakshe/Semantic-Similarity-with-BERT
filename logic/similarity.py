"""
    Created by KaushiRajapakshe on 15/08/2021.
"""
# Importing all required libraries to work with the dataset and model
import numpy as np
from bert.BertSemanticDataGenerator import BertSemanticDataGenerator
from constant.constant import labels


# Inference on custom sentencesÒ
class Similarity:

    def check_similarity(self, sentence1, sentence2):
        sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
        test_data = BertSemanticDataGenerator(
            sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
        )

        proba = self.model.predict(test_data)[0]
        idx = np.argmax(proba)
        proba = f"{proba[idx]: .2f}%"
        pred = labels[idx]
        return pred, proba
