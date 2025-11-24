import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report

# --- Configuration ---
MODEL_NAME = "aubmindlab/bert-base-arabertv02" # A popular Arabic BERT model
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3

class BERT_NER:
    """
    BERT-based model for Named Entity Recognition (Token Classification).
    Fine-tunes a pre-trained Arabic BERT model (AraBERT) on the NER task.
    """
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=None) # num_labels will be set during training
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_data(self, sentences):
        """
        Convert raw sentences into Hugging Face Dataset format, tokenizing and aligning labels.
        """
        words = [[item[0] for item in sent] for sent in sentences]
        tags = [[item[1] for item in sent] for sent in sentences]
        
        # 1. Get all unique tags and create mappings
        unique_tags = sorted(list(set(tag for sent_tags in tags for tag in sent_tags)))
        self.tag_to_id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id_to_tag = {i: tag for tag, i in self.tag_to_id.items()}
        self.model.config.id2label = self.id_to_tag
        self.model.config.label2id = self.tag_to_id
        self.model.num_labels = len(unique_tags)
        
        # 2. Convert tags to IDs
        label_ids = [[self.tag_to_id[tag] for tag in sent_tags] for sent_tags in tags]
        
        # 3. Tokenize and align labels
        tokenized_inputs = self.tokenizer(words, truncation=True, is_split_into_words=True, max_length=MAX_LENGTH, padding="max_length")
        
        new_labels = []
        for i, labels in enumerate(label_ids):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids_aligned = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids_aligned.append(-100) # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids_aligned.append(labels[word_idx]) # Start of a new word
                else:
                    label_ids_aligned.append(labels[word_idx]) # Continuation of a word (subword token)
                previous_word_idx = word_idx
            new_labels.append(label_ids_aligned)
            
        tokenized_inputs["labels"] = new_labels
        
        # Convert to Hugging Face Dataset
        dataset_dict = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_inputs["labels"],
        }
        
        return Dataset.from_dict(dataset_dict), words, tags

    def train(self, train_sentences, eval_sentences=None):
        """
        Fine-tune the BERT model.
        :param train_sentences: list of training sentences.
        :param eval_sentences: list of evaluation sentences.
        """
        print("Preparing BERT-NER data...")
        train_dataset, _, _ = self._prepare_data(train_sentences)
        
        if eval_sentences:
            eval_dataset, _, _ = self._prepare_data(eval_sentences)
        else:
            eval_dataset = None
            
        print("Starting BERT-NER fine-tuning...")
        
        training_args = TrainingArguments(
            output_dir="./bert_ner_results",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./bert_ner_logs',
            logging_steps=10,
            evaluation_strategy="epoch" if eval_sentences else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_sentences else False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        print("BERT-NER fine-tuning complete.")

    def _compute_metrics(self, p):
        """Compute metrics for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[self.id_to_tag[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id_to_tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Align predictions to true labels (needed because of tokenization alignment)
        # This is a simplified approach; a proper evaluation requires re-aligning to words.
        # For this project, we will use seqeval on the token-level predictions after filtering -100.
        
        # Since the Trainer's compute_metrics works on token-level, we'll return a simple accuracy
        # and rely on the full evaluation function for the final report.
        
        # Flatten lists for simple accuracy
        flat_true = [item for sublist in true_labels for item in sublist]
        flat_pred = [item for sublist in true_predictions for item in sublist]
        
        accuracy = np.sum(np.array(flat_true) == np.array(flat_pred)) / len(flat_true)
        
        return {"accuracy": accuracy}

    def predict(self, sentences):
        """
        Predict tags for a list of sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: list of predicted tag sequences (list of lists of strings).
        """
        # 1. Tokenize and align
        words = [[item[0] for item in sent] for sent in sentences]
        
        tokenized_inputs = self.tokenizer(words, truncation=True, is_split_into_words=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
        
        # 2. Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in tokenized_inputs.items()})
        
        predictions = outputs.logits.argmax(dim=2).cpu().numpy()
        
        # 3. Convert IDs back to tags and align to words
        all_predicted_tags = []
        for i, (prediction, input_ids) in enumerate(zip(predictions, tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            predicted_tags = []
            previous_word_idx = None
            
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                
                if word_idx != previous_word_idx:
                    # This is the first token of a word, use its prediction
                    tag_id = prediction[token_idx]
                    tag = self.id_to_tag.get(tag_id, 'O') # Default to 'O' if tag is unknown
                    predicted_tags.append(tag)
                
                previous_word_idx = word_idx
            
            all_predicted_tags.append(predicted_tags)
            
        return all_predicted_tags

    def evaluate(self, sentences):
        """
        Evaluate the model on a list of annotated sentences.
        :param sentences: list of sentences, each a list of (word, tag) tuples.
        :returns: classification report string.
        """
        y_true = [[tag for _, tag in sent] for sent in sentences]
        y_pred = self.predict(sentences)
        
        # Ensure y_true and y_pred have the same structure (list of lists of tags)
        # and that inner lists have the same length (number of words in the sentence)
        
        # The predict function aligns predictions to words, so we can use seqeval directly.
        return classification_report(y_true, y_pred, zero_division=0)

if __name__ == '__main__':
    from scripts.data_utils import load_arabic_ner_data
    from sklearn.model_selection import train_test_split
    
    # Load data (BERT requires a lot of data, so we'll use a small subset for quick testing)
    sentences = load_arabic_ner_data()
    
    if not sentences:
        print("Could not load NER data. Exiting BERT-NER test.")
    else:
        # Use a small subset for quick testing
        sentences = sentences[:100] 
        
        # Split data
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Train and evaluate
        bert_model = BERT_NER()
        # We pass a small eval set to the trainer for logging, but the final report uses the full evaluate function
        bert_model.train(train_sentences, test_sentences) 
        
        print("\n--- BERT-NER Evaluation ---")
        report = bert_model.evaluate(test_sentences)
        print(report)
        
        # Example prediction
        example_sentence = [("و", "O"), ("قال", "O"), ("الرئيس", "B-PER"), ("التنفيذي", "I-PER"), ("ل", "I-PER"), ("شركة", "I-PER"), ("مرسيدس", "I-PER"), ("بنز", "I-PER"), ("،", "O"), ("ديتر", "I-PER"), ("زيتشه", "I-PER"), ("،", "O"), ("إن", "O"), ("الشركة", "O"), ("تتوقع", "O"), ("نموا", "O"), ("قويا", "O"), ("في", "O"), ("الربع", "B-TIM"), ("الرابع", "I-TIM"), (".", "O")]
        predicted_tags = bert_model.predict([example_sentence])[0]
        print("\n--- BERT-NER Example Prediction ---")
        print(f"Sentence: {[word for word, _ in example_sentence]}")
        print(f"Predicted Tags: {predicted_tags}")
