import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report

class CRF_NER:
    """
    Conditional Random Field (CRF) model for Named Entity Recognition.
    CRF is a discriminative sequence model that uses rich feature engineering.
    """
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def word2features(self, sent, i):
        """
        Extract features for a word at index i in a sentence.
        Features are crucial for CRF performance.
        """
        word = sent[i][0]
        
        features = {
            'bias': 1.0,
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
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True # Beginning of Sentence

        if i < len(sent) - 1:
            word1 = sent[i+1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True # End of Sentence

        return features

    def sent2features(self, sent):
        """Convert a sentence (list of (word, tag) tuples) into a list of feature dictionaries."""
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        """Convert a sentence (list of (word, tag) tuples) into a list of tags."""
        return [tag for word, tag in sent]

    def train(self, sentences):
        """
        Train the CRF model.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        """
        print("Training CRF-NER model...")
        X_train = [self.sent2features(s) for s in sentences]
        y_train = [self.sent2labels(s) for s in sentences]
        
        self.model.fit(X_train, y_train)
        print("CRF-NER model trained.")

    def predict(self, sentences):
        """
        Predict tags for a list of sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: list of predicted tag sequences.
        """
        X_test = [self.sent2features(s) for s in sentences]
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, sentences):
        """
        Evaluate the model on a list of annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: classification report string.
        """
        y_true = [self.sent2labels(s) for s in sentences]
        y_pred = self.predict(sentences)
        
        # seqeval is the standard for NER evaluation
        return classification_report(y_true, y_pred, zero_division=0)

if __name__ == '__main__':
    from scripts.data_utils import load_arabic_ner_data
    
    # Load data (assuming data files are in place)
    sentences = load_arabic_ner_data()
    
    if not sentences:
        print("Could not load NER data. Exiting CRF-NER test.")
    else:
        # Split data
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Train and evaluate
        crf_model = CRF_NER()
        crf_model.train(train_sentences)
        
        print("\n--- CRF-NER Evaluation ---")
        report = crf_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("و", "O"), ("قال", "O"), ("الرئيس", "B-PER"), ("التنفيذي", "I-PER"), ("ل", "I-PER"), ("شركة", "I-PER"), ("مرسيدس", "I-PER"), ("بنز", "I-PER"), ("،", "O"), ("ديتر", "I-PER"), ("زيتشه", "I-PER"), ("،", "O"), ("إن", "O"), ("الشركة", "O"), ("تتوقع", "O"), ("نموا", "O"), ("قويا", "O"), ("في", "O"), ("الربع", "B-TIM"), ("الرابع", "I-TIM"), (".", "O")]
        predicted_tags = crf_model.predict([example_sentence])[0]
        print("\n--- CRF-NER Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
