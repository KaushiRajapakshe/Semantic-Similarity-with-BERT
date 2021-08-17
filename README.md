# Semantic Similarity with BERT

**Natural Language Inference by fine-tuning BERT model on SNLI (Stanford Natural Language Inference) Corpus.**


BERT model that takes two sentences as inputs and that outputs a similarity score for these two sentences.
Dataset

- Download [dataset](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus)
- Save the downloaded csv file inside the Dataset Folder

**Dataset Overview:**
- sentence1: A man and a woman are sitting at a table outside, next to a small flower garden.
- sentence2: The woman is having lunch with another woman.
- similarity: This is the label chosen by the majority of annotators. 
  
Here are the "similarity" label values in the dataset:
- Contradiction: The sentences share no similarity.
- Entailment: The sentences have similar meaning.
- Neutral: The sentences are neutral. 
  
**Setup**
* Package requirements

    ```!pip install numpy==1.19.2```

    ```!pip install pandas==1.1.3```

    ```!pip install tensorflow==2.6.0```

    ```!pip install transformers==2.11.0```
  
**References**

* [Mohamad Merchant - Keras](https://keras.io/examples/nlp/semantic_similarity_with_bert/)
* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)

