# Model file name
filename = 'finalized_bert_model1.sav'
# Check results on some example sentence pairs - 1
sentence1 = "The young boys are playing outdoors and the man is smiling nearby."
sentence2 = "The kids are playing outdoors near a man with a smile."
# Check results on some example sentence pairs - 2
sentence3 = "A group of children is playing in the house and there is no man standing in the background."
sentence4 = "A group of kids is playing in a yard and an old man is standing in the background."
# Check results on some example sentence pairs - 3
sentence5 = "Two people are kickboxing and spectators are not watching."
sentence6 = "Two people are kickboxing and spectators are watching."
# Model configuration
# Maximum length of input sentence to the model.
max_length = 128
batch_size = 32
epochs = 2
dataset_train = "dataset/snli_1.0_train.csv"
dataset_dev = "dataset/snli_1.0_dev.csv"
dataset_test = "dataset/snli_1.0_test.csv"
# Labels in the dataset.
labels = ["contradiction", "entailment", "neutral"]
initial_state = "initial_state"
app_config_path = "app.ini"
