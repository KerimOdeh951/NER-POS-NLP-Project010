import os
from sklearn.model_selection import train_test_split
from data_utils import load_arabic_pos_data
from POS_models.rule_based_pos import RuleBasedPOS
from POS_models.maxent_pos import MaxEntPOS
from POS_models.bilstm_pos import BiLSTM_POS

def train_and_evaluate_pos_models():
    """
    Loads POS data, splits it, trains all three POS models, and saves the evaluation reports.
    """
    print("--- Starting POS Models Training and Evaluation ---")
    
    # 1. Load Data
    sentences = load_arabic_pos_data()
    
    if not sentences:
        print("Error: No POS data loaded. Cannot proceed with training.")
        return

    # 2. Split Data (80% Train, 20% Test)
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
    print(f"\nTotal POS sentences: {len(sentences)}")
    print(f"Training sentences: {len(train_sentences)}")
    print(f"Testing sentences: {len(test_sentences)}")
    
    # Create outputs directory
    os.makedirs("../outputs", exist_ok=True)
    
    # --- Rule-Based POS ---
    print("\n--- Training Rule-Based POS ---")
    rb_model = RuleBasedPOS()
    rb_model.train(train_sentences)
    report_rb = rb_model.evaluate(test_sentences)
    
    with open("../outputs/pos_rule_results.txt", "w", encoding="utf-8") as f:
        f.write("Rule-Based POS Evaluation Report:\n")
        f.write(report_rb)
    print("Rule-Based POS report saved to outputs/pos_rule_results.txt")
    
    # --- MaxEnt POS ---
    print("\n--- Training MaxEnt POS ---")
    maxent_model = MaxEntPOS()
    maxent_model.train(train_sentences)
    report_maxent = maxent_model.evaluate(test_sentences)
    
    with open("../outputs/pos_maxent_results.txt", "w", encoding="utf-8") as f:
        f.write("MaxEnt POS Evaluation Report:\n")
        f.write(report_maxent)
    print("MaxEnt POS report saved to outputs/pos_maxent_results.txt")
    
    # --- BiLSTM POS ---
    # Note: BiLSTM training is resource-intensive and slow. We use a small subset for quick demonstration.
    print("\n--- Training BiLSTM POS (using a small subset for efficiency) ---")
    
    # Use a smaller subset of the data for BiLSTM training to avoid long runtimes
    bilstm_train_sentences = train_sentences[:500] 
    bilstm_test_sentences = test_sentences[:100]
    
    bilstm_model = BiLSTM_POS()
    # We pass a small eval set to the trainer for logging, but the final report uses the full evaluate function
    bilstm_model.train(bilstm_train_sentences, bilstm_test_sentences)
    report_bilstm = bilstm_model.evaluate(bilstm_test_sentences)
    
    with open("../outputs/pos_bilstm_results.txt", "w", encoding="utf-8") as f:
        f.write("BiLSTM POS Evaluation Report:\n")
        f.write(report_bilstm)
    print("BiLSTM POS report saved to outputs/pos_bilstm_results.txt")
    
    print("\n--- POS Models Training and Evaluation Complete ---")

if __name__ == '__main__':
    # Change directory to the project root to ensure correct relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    train_and_evaluate_pos_models()
