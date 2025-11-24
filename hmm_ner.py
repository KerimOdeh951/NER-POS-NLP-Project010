import math
from collections import defaultdict
from seqeval.metrics import classification_report

class HMM_NER:
    """
    Hidden Markov Model for Named Entity Recognition.
    This model treats NER tags as hidden states and words as observable outputs.
    It uses the Viterbi algorithm for inference.
    """
    def __init__(self):
        self.tags = set()
        self.start_counts = defaultdict(int)
        self.trans_counts = defaultdict(lambda: defaultdict(int))
        self.emit_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.smoothing_param = 1e-5 # Laplace smoothing

    def train(self, sentences):
        """
        Train HMM parameters (start, transition, emission probabilities) from annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        """
        print("Training HMM-NER model...")
        for sent in sentences:
            prev_tag = None
            for i, (word, tag) in enumerate(sent):
                self.vocab.add(word)
                self.tags.add(tag)
                
                # Count tag occurrences
                self.tag_counts[tag] += 1
                
                # Count emission
                self.emit_counts[tag][word] += 1
                
                if i == 0:
                    # Count start tag
                    self.start_counts[tag] += 1
                else:
                    # Count transition
                    self.trans_counts[prev_tag][tag] += 1
                
                prev_tag = tag
        
        # Convert counts to probabilities (with Laplace smoothing)
        self.start_prob = {tag: (count + self.smoothing_param) / (sum(self.start_counts.values()) + len(self.tags) * self.smoothing_param) 
                           for tag, count in self.start_counts.items()}
        
        self.trans_prob = {
            prev_tag: {
                tag: (count + self.smoothing_param) / (self.tag_counts[prev_tag] + len(self.tags) * self.smoothing_param)
                for tag, count in counts.items()
            }
            for prev_tag, counts in self.trans_counts.items()
        }
        
        self.emit_prob = {
            tag: {
                word: (count + self.smoothing_param) / (self.tag_counts[tag] + len(self.vocab) * self.smoothing_param)
                for word, count in counts.items()
            }
            for tag, counts in self.emit_counts.items()
        }
        print("HMM-NER model trained.")

    def viterbi(self, sentence):
        """
        Viterbi algorithm to find the most likely sequence of tags for a given sentence.
        :param sentence: list of words.
        :returns: list of predicted tags.
        """
        if not self.tags:
            return ['O'] * len(sentence)
            
        words = [word for word, _ in sentence]
        T = len(words)
        tags = list(self.tags)
        N = len(tags)
        
        # Viterbi matrix: V[t][i] is the probability of the most likely tag sequence 
        # ending at time t with tag i.
        V = [{}] * T
        # Backpointer matrix: P[t][i] stores the index of the previous tag.
        P = [{}] * T
        
        # 1. Initialization
        for i, tag in enumerate(tags):
            word = words[0]
            # Emission probability: P(word | tag)
            emit = self.emit_prob.get(tag, {}).get(word, self.smoothing_param / (self.tag_counts.get(tag, 0) + len(self.vocab) * self.smoothing_param))
            # Start probability: P(tag at start)
            start = self.start_prob.get(tag, self.smoothing_param / (sum(self.start_counts.values()) + N * self.smoothing_param))
            
            V[0][tag] = math.log(start) + math.log(emit)
            P[0][tag] = None
            
        # 2. Recursion
        for t in range(1, T):
            V[t] = {}
            P[t] = {}
            word = words[t]
            for i, tag in enumerate(tags):
                max_prob = -math.inf
                best_prev_tag = None
                
                # Emission probability: P(word | tag)
                emit = self.emit_prob.get(tag, {}).get(word, self.smoothing_param / (self.tag_counts.get(tag, 0) + len(self.vocab) * self.smoothing_param))
                log_emit = math.log(emit)
                
                for j, prev_tag in enumerate(tags):
                    # Transition probability: P(tag | prev_tag)
                    trans = self.trans_prob.get(prev_tag, {}).get(tag, self.smoothing_param / (self.tag_counts.get(prev_tag, 0) + N * self.smoothing_param))
                    log_trans = math.log(trans)
                    
                    prob = V[t-1][prev_tag] + log_trans + log_emit
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag
                
                V[t][tag] = max_prob
                P[t][tag] = best_prev_tag
                
        # 3. Termination and Path Backtracking
        best_path_prob = -math.inf
        last_tag = None
        
        for tag, prob in V[T-1].items():
            if prob > best_path_prob:
                best_path_prob = prob
                last_tag = tag
                
        best_path = [last_tag]
        curr_tag = last_tag
        for t in range(T-1, 0, -1):
            curr_tag = P[t][curr_tag]
            best_path.insert(0, curr_tag)
            
        return best_path

    def evaluate(self, sentences):
        """
        Evaluate the model on a list of annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: classification report string.
        """
        y_true = []
        y_pred = []
        
        for sentence in sentences:
            true_tags = [tag for _, tag in sentence]
            predicted_tags = self.viterbi(sentence)
            
            y_true.append(true_tags)
            y_pred.append(predicted_tags)
            
        return classification_report(y_true, y_pred, zero_division=0)

if __name__ == '__main__':
    from scripts.data_utils import load_arabic_ner_data
    
    # Load data (assuming data files are in place)
    sentences = load_arabic_ner_data()
    
    if not sentences:
        print("Could not load NER data. Exiting HMM-NER test.")
    else:
        # Simple split for demonstration: 80% train, 20% test
        split_point = int(0.8 * len(sentences))
        train_sentences = sentences[:split_point]
        test_sentences = sentences[split_point:]
        
        # Train and evaluate
        hmm_model = HMM_NER()
        hmm_model.train(train_sentences)
        
        print("\n--- HMM-NER Evaluation ---")
        report = hmm_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("أعلن", "O"), ("اتحاد", "B-ORG"), ("صناعة", "I-ORG"), ("السيارات", "I-ORG"), ("في", "O"), ("ألمانيا", "B-LOC"), ("امس", "O"), ("الاول", "O"), (".", "O")]
        predicted_tags = hmm_model.viterbi(example_sentence)
        print("\n--- HMM-NER Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
