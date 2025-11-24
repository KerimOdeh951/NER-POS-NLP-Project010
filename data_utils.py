import pandas as pd
import os

# --- Constants ---
NER_DATA_PATH = "data/ANERCorp.conll"
POS_DATA_PATH = "data/ar_padt-ud-train.conllu"

def load_conll_data(file_path, tag_column_index):
    """
    General function to load data from CoNLL-like format.
    :param file_path: Path to the CoNLL file.
    :param tag_column_index: Index of the column containing the required tag (e.g., 1 for NER, 3 for UPOS in CoNLL-U).
    :returns: list of sentences, each sentence is a list of (word, tag) tuples.
    """
    sentences = []
    current_sentence = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return []
        
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                # Sentence break (empty line or comment)
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
                
            parts = line.split('\t')
            
            if len(parts) > tag_column_index:
                # CoNLL-U columns: ID, FORM, LEMMA, UPOS, XPOS, ...
                # CoNLL-NER columns: WORD, TAG
                
                # For CoNLL-U (POS), skip multi-word token lines (ID contains '-')
                if tag_column_index == 3 and '-' in parts[0]:
                    continue
                    
                word = parts[0] if tag_column_index == 1 else parts[1] # Word is at index 0 for simple CoNLL-NER, or 1 for CoNLL-U
                tag = parts[tag_column_index]
                
                current_sentence.append((word, tag))
                
    # append last sentence if not empty
    if current_sentence:
        sentences.append(current_sentence)
        
    print(f"Successfully loaded {len(sentences)} sentences from {file_path}.")
    return sentences

def load_arabic_ner_data(file_path=NER_DATA_PATH):
    """
    Load Arabic NER data from local CoNLL file.
    ANERCorp format: Word \t Tag (assuming simple 2-column CoNLL format)
    """
    # Assuming ANERCorp is a simple 2-column CoNLL format: Word \t Tag
    # The tag is at index 1 (0-indexed)
    return load_conll_data(file_path, tag_column_index=1)

def load_arabic_pos_data(file_path=POS_DATA_PATH):
    """
    Load Arabic POS data from local CoNLL-U file.
    UD_Arabic-PADT format: ID \t FORM \t LEMMA \t UPOS \t XPOS \t ...
    The UPOS tag is at index 3 (0-indexed)
    """
    return load_conll_data(file_path, tag_column_index=3)

if __name__ == '__main__':
    # Test data loading
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Test NER data loading
    ner_sentences = load_arabic_ner_data()
    print(f"\nNER Data Sample (First Sentence): {ner_sentences[0] if ner_sentences else 'N/A'}")
    
    # Test POS data loading
    pos_sentences = load_arabic_pos_data()
    print(f"\nPOS Data Sample (First Sentence): {pos_sentences[0] if pos_sentences else 'N/A'}")
