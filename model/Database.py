"""
    Created by KaushiRajapakshe on 27/08/2021.

    Database class

"""


class Database:

    def __init__(self):
        self.db = ''
        self.sentence_dict = {}
        self.sentence_id_list = []
        self.sentence_id = ''

    # getter method db
    def get_db(self):
        return self.db

    # setter method db
    def set_db(self, db):
        self.db = db

    # getter method sentence dict
    def get_sentence_dict(self):
        return self.sentence_dict

    # setter method sentence dict
    def set_sentence_dict(self, sentence_dict):
        self.sentence_dict = sentence_dict

    # getter method sentence id list
    def get_sentence_id_list(self):
        return self.sentence_id_list

    # setter method sentence id list
    def set_sentence_id_list(self, sentence_id_list):
        self.sentence_id_list = sentence_id_list

    # getter method sentence id
    def get_sentence_id(self):
        return self.sentence_id

    # setter method sentence id
    def set_sentence_id(self, sentence_id):
        self.sentence_id = sentence_id
