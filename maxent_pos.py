from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

class MaxEntPOS:
    """
    Maximum Entropy (MaxEnt) Tagger for Part-of-Speech Tagging.
    Implemented using Logistic Regression (a MaxEnt model) with feature engineering.
    """
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100, verbose=0)
        self.tags = []
        self.tag_to_id = {}
        self.id_to_tag = {}

    def _word2features(self, sent, i):
        """
        Extract features for a word at index i in a sentence.
        Features include the word itself, its neighbors, and morphological features.
        """
        word = sent[i][0]
        
        features = {
            'bias': 1.0,
            'word': word,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.is_arabic()': all('\u0600' <= char <= '\u06FF' for char in word),
        }

        if i > 0:
            word1 = sent[i-1][0]
            features.update({
                '-1:word': word1,
                '-1:word.lower()': word1.lower(),
            })
        else:
            features['BOS'] = True # Beginning of Sentence

        if i < len(sent) - 1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word': word1,
                '+1:word.lower()': word1.lower(),
            })
        else:
            features['EOS'] = True # End of Sentence

        return features

    def _sent2features(self, sent):
        """Convert a sentence (list of (word, tag) tuples) into a list of feature dictionaries."""
        return [self._word2features(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        """Convert a sentence (list of (word, tag) tuples) into a list of tags."""
        return [tag for word, tag in sent]

    def train(self, sentences):
        """
        Train the MaxEnt (Logistic Regression) model.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        """
        print("Training MaxEnt-POS model...")
        
        # 1. Extract features and labels
        X_train_features = [f for s in sentences for f in self._sent2features(s)]
        y_train_labels = [l for s in sentences for l in self._sent2labels(s)]
        
        # 2. Vectorize features
        X_train = self.vectorizer.fit_transform(X_train_features)
        
        # 3. Train the model
        self.model.fit(X_train, y_train_labels)
        
        # Store tags for later use
        self.tags = self.model.classes_
        self.tag_to_id = {tag: i for i, tag in enumerate(self.tags)}
        self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}
        
        print("MaxEnt-POS model trained.")

    def predict(self, sentences):
        """
        Predict tags for a list of sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: list of predicted tag sequences (list of lists of strings).
        """
        y_pred = []
        for sent in sentences:
            X_test_features = self._sent2features(sent)
            X_test = self.vectorizer.transform(X_test_features)
            predicted_labels = self.model.predict(X_test)
            y_pred.append(list(predicted_labels))
        return y_pred

    def evaluate(self, sentences):
        """
        Evaluate the model on a list of annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: classification report string.
        """
        y_true = [self._sent2labels(s) for s in sentences]
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
        print("Could not load POS data. Exiting MaxEnt-POS test.")
    else:
        # Split data
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Train and evaluate
        maxent_model = MaxEntPOS()
        maxent_model.train(train_sentences)
        
        print("\n--- MaxEnt-POS Evaluation ---")
        report = maxent_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("الرئيس", "NOUN"), ("يزور", "VERB"), ("المملكة", "NOUN"), ("العربية", "ADJ"), ("السعودية", "NOUN"), ("غدا", "ADV"), (".", "PUNCT")]
        predicted_tags = maxent_model.predict([example_sentence])[0]
        print("\n--- MaxEnt-POS Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
