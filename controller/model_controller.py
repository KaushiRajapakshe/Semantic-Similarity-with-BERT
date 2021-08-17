"""
    Created by KaushiRajapakshe on 15/08/2021.

    You can save the trained model to disk and load the model using pickle
"""

# Importing all required libraries to save model
import pickle
import tensorflow as tf
import transformers

from bert.BertSemanticDataGenerator import BertSemanticDataGenerator
from constant.constant import filename, max_length, batch_size, epochs


class ModelController:

    # Build the model
    # Create the model under a distribution strategy scope.
    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            # Encoded token ids from BERT tokenizer.
            input_ids = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="input_ids"
            )
            # Attention masks indicates to the model which tokens should be attended to.
            attention_masks = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="attention_masks"
            )
            # Token type ids are binary masks identifying different sequences in the model.
            token_type_ids = tf.keras.layers.Input(
                shape=(max_length,), dtype=tf.int32, name="token_type_ids"
            )
            # Loading pretrained BERT model.
            self.set_bert_model(transformers.TFBertModel.from_pretrained("bert-base-uncased"))
            # Freeze the BERT model to reuse the pretrained features without modifying them.
            self.bert_model.trainable = False

            sequence_output, pooled_output = self.bert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
            # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            )(sequence_output)
            # Applying hybrid pooling approach to bi_lstm sequence output.
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(0.3)(concat)
            output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
            self.set_model(tf.keras.models.Model(
                inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            ))

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["acc"],
            )

        print(f"Strategy: {strategy}")
        self.model.summary()

    # Create train and validation data generators
    def data_generation(self):
        self.set_train_data(BertSemanticDataGenerator(
            self.train_df[["sentence1", "sentence2"]].values.astype("str"),
            self.y_train,
            batch_size=batch_size,
            shuffle=True,
        ))
        self.set_valid_data(BertSemanticDataGenerator(
            self.valid_df[["sentence1", "sentence2"]].values.astype("str"),
            self.y_val,
            batch_size=batch_size,
            shuffle=False,
        ))

    """
    Train the Model
        Training is done only for the top layers to perform "feature extraction",
        which will allow the model to use the representations of the pretrained model.
    """
    def train_model(self):
        history = self.model.model.fit(
            self.dataset.train_data,
            validation_data=self.dataset.valid_data,
            epochs=epochs,
            use_multiprocessing=True,
            workers=-1,
        )

    """
    Fine-tuning
        This step must only be performed after the feature extraction model has
        been trained to convergence on the new data.
        This is an optional last step where `bert_model` is unfreezed and retrained
        with a very low learning rate. This can deliver meaningful improvement by
        incrementally adapting the pretrained features to the new data.
    """
    # Unfreeze the bert_model
    def fine_tuning_model(self):
        self.model.bert_model.trainable = True
        # Recompile the model to make the change effective.
        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.model.summary()

        # Train the entire model end-to-end
        history = self.model.model.fit(
            self.dataset.train_data,
            validation_data=self.dataset.valid_data,
            epochs=epochs,
            use_multiprocessing=True,
            workers=-1,
        )

        # Evaluate model on the test set
        test_data = BertSemanticDataGenerator(
            self.dataset.test_df[["sentence1", "sentence2"]].values.astype("str"),
            self.dataset.y_test,
            batch_size=batch_size,
            shuffle=False,
        )
        self.model.model.evaluate(test_data, verbose=1)

    # save the model to disk
    def save_model(self):
        pickle.dump(self.model, open(filename, 'wb'))


def get_load_model(self):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
