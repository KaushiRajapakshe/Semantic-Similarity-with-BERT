"""
    Created by KaushiRajapakshe on 17/08/2021.

    Semantic Similarity with BERT (Bidirectional Encoder Representations from Transformers)
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

    Main Controller
"""


# Importing all required libraries to work with the main controller
from bert.BertSemanticDataGenerator import BertSemanticDataGenerator
from constant.constant import sentence1, sentence2, sentence3, sentence4, sentence5, sentence6, initial_state, \
    app_config_path
from controller import config_controller
from controller.database_controller import DatabaseController
from controller.dataset_controller import DatasetController
from controller.model_controller import ModelController
from logic.firebase_database import Database
from logic.similarity import Similarity


class MainController:

    def main(self):

        # initialise config object using the config_controller
        app_config = config_controller.init_config(app_config_path)

        # get string value initial_state
        initial_state_value = app_config.get('default', initial_state)

        # initialize firebase
        Database.init_firebase(self.database)

        # initial_state == enable
        if initial_state_value == 'enable':
            DatasetController.data_shape(self.dataset)
            DatasetController.display_sample_dataset(self.dataset)
            DatasetController.data_preprocessing(self.dataset)
            DatasetController.dis_train_target(self.dataset)
            DatasetController.dis_validation_target(self.dataset)
            DatasetController.skip_hyphen_values(self.dataset)
            DatasetController.encode(self.dataset)

            BertSemanticDataGenerator

            ModelController.build_model(self.model)
            ModelController.data_generation(self.dataset)
            ModelController.train_model(self)
            ModelController.fine_tuning_model(self)
            # ModelController.save_model(self.model)
        else:
            print('else')

        DatabaseController.get_sentence_score(self.database)
        score1 = Similarity.check_similarity(self.model, sentence1, sentence2)
        print(score1)
        DatabaseController.save_sentence_score(self.database, sentence1, sentence2, score1)
        score2 = Similarity.check_similarity(self.model, sentence3, sentence4)
        print(score2)
        DatabaseController.save_sentence_score(self.database, sentence3, sentence4, score2)
        score3 = Similarity.check_similarity(self.model, sentence5, sentence6)
        print(score3)
        DatabaseController.save_sentence_score(self.database, sentence5, sentence6, score3)
