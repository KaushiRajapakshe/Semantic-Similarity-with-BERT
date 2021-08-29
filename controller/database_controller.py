"""
    Created by KaushiRajapakshe on 28/08/2021.

    Database Controller
"""


# Importing all required libraries to database controller
class DatabaseController:

    # Save Sentences after checking similarity with similarity score
    def save_sentence_score(self, sentence1, sentence2, score):
        self.set_sentence_id(self.get_sentence_id()+1)
        doc_ref = self.db.collection(u'sentenceScore').document(u'sentence_doc_'+str(self.get_sentence_id()))
        doc_ref.set({
            u'id': self.get_sentence_id(),
            u'sentence_01': sentence1,
            u'sentence_02': sentence2,
            u'score': score
        })

    # Get all Sentences with similarity score
    def get_sentence_score(self):
        sentence_ref = self.db.collection(u'sentenceScore')
        docs = sentence_ref.stream()

        for doc in docs:
            self.sentence_dict[doc.id] = doc.to_dict()
            self.sentence_id_list.append(self.sentence_dict[doc.id].get('id'))
            # print(f'{doc.id} => {doc.to_dict()}')

        # Sort the sentence id list
        self.get_sentence_id_list().sort()

        self.set_sentence_id(self.sentence_id_list[len(self.sentence_id_list) - 1])
        return self.sentence_dict
