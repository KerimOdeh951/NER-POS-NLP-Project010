import os
from sklearn.model_selection import train_test_split
from data_utils import load_arabic_ner_data
from NER_models.hmm_ner import HMM_NER
from NER_models.crf_ner import CRF_NER
from NER_models.bert_ner import BERT_NER

def train_and_evaluate_ner_models():
    """
    Loads NER data, splits it, trains all three NER models, and saves the evaluation reports.
    """
    print("--- Starting NER Models Training and Evaluation ---")
    
    # 1. Load Data
    sentences = load_arabic_ner_data()
    
    if not sentences:
        print("Error: No NER data loaded. Cannot proceed with training.")
        return

    # 2. Split Data (80% Train, 20% Test)
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
    print(f"\nTotal NER sentences: {len(sentences)}")
    print(f"Training sentences: {len(train_sentences)}")
    print(f"Testing sentences: {len(test_sentences)}")
    
    # Create outputs directory
    os.makedirs("../outputs", exist_ok=True)
    
    # --- HMM-NER ---
    print("\n--- Training HMM-NER ---")
    hmm_model = HMM_NER()
    hmm_model.train(train_sentences)
    report_hmm = hmm_model.evaluate(test_sentences)
    
    with open("../outputs/ner_hmm_results.txt", "w", encoding="utf-8") as f:
        f.write("HMM-NER Evaluation Report:\n")
        f.write(report_hmm)
    print("HMM-NER report saved to outputs/ner_hmm_results.txt")
    
    # --- CRF-NER ---
    print("\n--- Training CRF-NER ---")
    crf_model = CRF_NER()
    crf_model.train(train_sentences)
    report_crf = crf_model.evaluate(test_sentences)
    
    with open("../outputs/ner_crf_results.txt", "w", encoding="utf-8") as f:
        f.write("CRF-NER Evaluation Report:\n")
        f.write(report_crf)
    print("CRF-NER report saved to outputs/ner_crf_results.txt")
    
    # --- BERT-NER ---
    # Note: BERT training is resource-intensive and slow. We use a small subset for quick demonstration.
    print("\n--- Training BERT-NER (using a small subset for efficiency) ---")
    
    # Use a smaller subset of the data for BERT training to avoid long runtimes
    bert_train_sentences = train_sentences[:500] 
    bert_test_sentences = test_sentences[:100]
    
    bert_model = BERT_NER()
    bert_model.train(bert_train_sentences, bert_test_sentences)
    report_bert = bert_model.evaluate(bert_test_sentences)
    
    with open("../outputs/ner_bert_results.txt", "w", encoding="utf-8") as f:
        f.write("BERT-NER Evaluation Report:\n")
        f.write(report_bert)
    print("BERT-NER report saved to outputs/ner_bert_results.txt")
    
    print("\n--- NER Models Training and Evaluation Complete ---")

if __name__ == '__main__':
    # Change directory to the project root to ensure correct relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    train_and_evaluate_ner_models()
