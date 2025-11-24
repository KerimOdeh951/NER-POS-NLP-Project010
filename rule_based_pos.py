import re
from collections import defaultdict
from sklearn.metrics import classification_report

class RuleBasedPOS:
    """
    Rule-Based Part-of-Speech Tagger for Arabic.
    Uses simple rules (suffixes, prefixes, common words) as a baseline.
    """
    def __init__(self):
        # A simple lexicon for common words (can be expanded)
        self.lexicon = {
            "الذي": "PRON", "التي": "PRON", "هذا": "PRON", "هذه": "PRON",
            "في": "ADP", "من": "ADP", "إلى": "ADP", "على": "ADP", "ب": "ADP", "ل": "ADP",
            "و": "CCONJ", "ف": "CCONJ", "ثم": "CCONJ",
            "أن": "SCONJ", "إن": "SCONJ", "قد": "PART",
            "كان": "VERB", "يكون": "VERB", "قال": "VERB", "يقول": "VERB",
            "الله": "NOUN", "الرئيس": "NOUN", "الشركة": "NOUN", "اليوم": "NOUN",
            "كبير": "ADJ", "جديد": "ADJ", "عربي": "ADJ",
        }
        
        # Common Arabic prefixes and suffixes for rule-based tagging
        self.prefixes = {
            "ال": "DET", # Definite article
            "و": "CCONJ", # Conjunction
            "ف": "CCONJ", # Conjunction
            "ب": "ADP", # Preposition
            "ل": "ADP", # Preposition
            "ك": "ADP", # Preposition
            "س": "PART", # Future particle
        }
        
        self.suffixes = {
            "ون": "NOUN", # Plural masculine
            "ين": "NOUN", # Plural masculine/oblique
            "ات": "NOUN", # Plural feminine
            "ة": "NOUN", # Feminine marker
            "ي": "PRON", # Possessive pronoun
            "نا": "PRON", # Possessive pronoun
        }
        
        # Default tag (most common tag in Arabic is NOUN or ADP/PUNCT)
        self.default_tag = "NOUN"

    def _apply_rules(self, word):
        """Apply rules to predict POS tag."""
        
        # 1. Lexicon lookup
        if word in self.lexicon:
            return self.lexicon[word]
        
        # 2. Punctuation/Number check
        if re.match(r'^\d+$', word):
            return "NUM"
        if re.match(r'^[.,;?!:()\[\]{}"]+$', word):
            return "PUNCT"
        
        # 3. Prefix check (longest match first)
        for prefix, tag in sorted(self.prefixes.items(), key=lambda item: len(item[0]), reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix):
                # Check if the remaining part is in the lexicon or is a verb/noun
                remaining_word = word[len(prefix):]
                if remaining_word in self.lexicon:
                    return self.lexicon[remaining_word]
                # Simple heuristic: if it starts with 'ال' (DET), the rest is likely NOUN or ADJ
                if prefix == "ال":
                    return "NOUN" # Default to NOUN after DET
                return tag # Return the prefix tag (e.g., ADP)
        
        # 4. Suffix check (longest match first)
        for suffix, tag in sorted(self.suffixes.items(), key=lambda item: len(item[0]), reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix):
                return tag
        
        # 5. Default tag
        return self.default_tag

    def train(self, sentences):
        """
        In a pure rule-based model, training is often just building the lexicon 
        and refining rules. Here, we'll just extract the lexicon from the data 
        for a slightly more data-driven approach.
        """
        print("Training Rule-Based POS model (building lexicon)...")
        for sent in sentences:
            for word, tag in sent:
                # Only add words that are not already in the lexicon and are not punctuation/numbers
                if word not in self.lexicon and not re.match(r'^[.,;?!:()\[\]{}"]+$', word) and not re.match(r'^\d+$', word):
                    self.lexicon[word] = tag
        print(f"Lexicon size: {len(self.lexicon)}")
        print("Rule-Based POS model ready.")

    def predict(self, sentences):
        """
        Predict POS tags for a list of sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: list of predicted tag sequences (list of lists of strings).
        """
        y_pred = []
        for sent in sentences:
            predicted_tags = [self._apply_rules(word) for word, _ in sent]
            y_pred.append(predicted_tags)
        return y_pred

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
    from sklearn.model_selection import train_test_split
    
    # Load data
    sentences = load_arabic_pos_data()
    
    if not sentences:
        print("Could not load POS data. Exiting Rule-Based POS test.")
    else:
        # Split data
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Train and evaluate
        rb_model = RuleBasedPOS()
        rb_model.train(train_sentences)
        
        print("\n--- Rule-Based POS Evaluation ---")
        report = rb_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("الرئيس", "NOUN"), ("يزور", "VERB"), ("المملكة", "NOUN"), ("العربية", "ADJ"), ("السعودية", "NOUN"), ("غدا", "ADV"), (".", "PUNCT")]
        predicted_tags = rb_model.predict([example_sentence])[0]
        print("\n--- Rule-Based POS Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
