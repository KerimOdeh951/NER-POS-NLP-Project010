import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
MAX_LEN = 50
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 5

class BiLSTM_POS:
    """
    Bidirectional LSTM (BiLSTM) model for Part-of-Speech Tagging.
    Uses an embedding layer and BiLSTM to capture sequential context.
    """
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.model = None
        self.vocab_size = 0
        self.tag_size = 0

    def _prepare_data(self, sentences):
        """
        Convert raw sentences into sequences of word IDs and tag IDs.
        """
        words = [[item[0] for item in sent] for sent in sentences]
        tags = [[item[1] for item in sent] for sent in sentences]
        
        # 1. Build vocabulary and tag set
        if not self.word_to_id:
            all_words = [word for sent in words for word in sent]
            unique_words = sorted(list(set(all_words)))
            self.word_to_id = {word: i + 2 for i, word in enumerate(unique_words)}
            self.word_to_id["PAD"] = 0
            self.word_to_id["UNK"] = 1
            self.id_to_word = {i: word for word, i in self.word_to_id.items()}
            self.vocab_size = len(self.word_to_id)
            
        if not self.tag_to_id:
            all_tags = [tag for sent in tags for tag in sent]
            unique_tags = sorted(list(set(all_tags)))
            self.tag_to_id = {tag: i + 1 for i, tag in enumerate(unique_tags)}
            self.tag_to_id["PAD"] = 0
            self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}
            self.tag_size = len(self.tag_to_id)
            
        # 2. Convert words and tags to IDs
        X = [[self.word_to_id.get(w, 1) for w in s] for s in words]
        y = [[self.tag_to_id.get(t, 0) for t in s] for s in tags]
        
        # 3. Pad sequences
        X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=0)
        y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=0)
        
        # 4. Convert tags to one-hot encoding
        y = [np.eye(self.tag_size)[i] for i in y]
        y = np.array(y)
        
        return X, y, tags

    def _build_model(self):
        """
        Build the BiLSTM model architecture.
        """
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
        self.model.add(Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=0.1)))
        self.model.add(TimeDistributed(Dense(self.tag_size, activation="softmax")))
        
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def train(self, train_sentences, eval_sentences=None):
        """
        Train the BiLSTM model.
        :param train_sentences: list of training sentences.
        :param eval_sentences: list of evaluation sentences.
        """
        print("Preparing BiLSTM-POS data...")
        X_train, y_train_onehot, y_train_tags = self._prepare_data(train_sentences)
        
        if eval_sentences:
            X_val, y_val_onehot, _ = self._prepare_data(eval_sentences)
            validation_data = (X_val, y_val_onehot)
        else:
            validation_data = None
            
        self._build_model()
        
        print("Starting BiLSTM-POS training...")
        self.model.fit(
            X_train, y_train_onehot, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            validation_data=validation_data,
            verbose=1
        )
        print("BiLSTM-POS training complete.")

    def predict(self, sentences):
        """
        Predict tags for a list of sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: list of predicted tag sequences (list of lists of strings).
        """
        X_test, _, _ = self._prepare_data(sentences)
        
        # Predict one-hot tags
        y_pred_onehot = self.model.predict(X_test, verbose=0)
        
        # Convert one-hot to tag IDs
        y_pred_ids = np.argmax(y_pred_onehot, axis=-1)
        
        # Convert tag IDs to tag strings, removing padding
        y_pred_tags = []
        for i, sentence in enumerate(sentences):
            pred_tags = []
            for j in range(len(sentence)):
                tag_id = y_pred_ids[i][j]
                tag = self.id_to_tag.get(tag_id, "UNK") # Default to UNK if ID is not found
                if tag != "PAD":
                    pred_tags.append(tag)
            y_pred_tags.append(pred_tags)
            
        return y_pred_tags

    def evaluate(self, sentences):
        """
        Evaluate the model on a list of annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: classification report string.
        """
        y_true = [[tag for _, tag in sent] for sent in sentences]
        y_pred = self.predict(sentences)
        
        # Flatten lists for token-level evaluation
        y_true_flat = [tag for sent in y_true for tag in sent]
        y_pred_flat = [tag for sent in y_pred for tag in sent]
        
        return classification_report(y_true_flat, y_pred_flat, zero_division=0)

if __name__ == '__main__':
    from scripts.data_utils import load_arabic_pos_data
    
    # Load data
    sentences = load_arabic_pos_data()
    
    if not sentences:
        print("Could not load POS data. Exiting BiLSTM-POS test.")
    else:
        # Split data
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Train and evaluate
        bilstm_model = BiLSTM_POS()
        bilstm_model.train(train_sentences, test_sentences)
        
        print("\n--- BiLSTM-POS Evaluation ---")
        report = bilstm_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("الرئيس", "NOUN"), ("يزور", "VERB"), ("المملكة", "NOUN"), ("العربية", "ADJ"), ("السعودية", "NOUN"), ("غدا", "ADV"), (".", "PUNCT")]
        predicted_tags = bilstm_model.predict([example_sentence])[0]
        print("\n--- BiLSTM-POS Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
