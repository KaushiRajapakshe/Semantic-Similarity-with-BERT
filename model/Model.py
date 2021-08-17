"""
    Created by KaushiRajapakshe on 15/08/2021.

    Model class

"""


class Model:

    def __init__(self):
        self.model = ''
        self.train_data = ''
        self.valid_data = ''
        self.bert_model = ''

    # getter method model
    def get_model(self):
        return self.model

    # setter method model
    def set_model(self, model):
        self.model = model

    # getter method train data
    def get_train_data(self):
        return self.train_data

    # setter method train data
    def set_train_data(self, train_data):
        self.train_data = train_data

    # getter method valid data
    def get_valid_data(self):
        return self.valid_data

    # setter method valid data
    def set_valid_data(self, valid_data):
        self.valid_data = valid_data

    # getter method bert model
    def get_bert_model(self):
        return self.bert_model

    # setter method bert model
    def set_bert_model(self, bert_model):
        self.bert_model = bert_model
