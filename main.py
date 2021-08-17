"""
    Created by KaushiRajapakshe on 15/08/2021.

    Semantic Similarity with BERT
    Natural Language Inference by fine-tuning BERT model on SNLI (Stanford Natural Language Inference) Corpus.

    * BERT model that takes two sentences as inputs and that outputs a similarity score for these two sentences.

    Setup
    * Package requirements
        - !pip install numpy==1.19.2
        - !pip install pandas==1.1.3
        - !pip install tensorflow==2.6.0
        - !pip install transformers==2.11.0

    References
    * Mohamad Merchant - https://keras.io/examples/nlp/semantic_similarity_with_bert/
    * [BERT](https://arxiv.org/pdf/1810.04805.pdf)
    * [SNLI](https://nlp.stanford.edu/projects/snli/)

"""

# Press the green button in the gutter to run the script.
from controller.main_controller import MainController
from logic.similarity import Similarity
from model.Dataset import Dataset
from model.Model import Model


class Main:
    if __name__ == '__main__':
        print('Welcome to BERT')

        def __init__(self):
            self.model = Model()
            self.dataset = Dataset()
            self.similarity = Similarity()


main = Main()
MainController.main(main)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
