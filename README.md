Arabic NER and POS Tagging Project

This project is a comprehensive Arabic Natural Language Processing (NLP) application focusing on two core tasks: Named Entity Recognition (NER) and Part-of-Speech Tagging (POS). The project implements six different models (three for each task) to provide a thorough comparison between classical and modern machine learning approaches.

Project Structure

The project is organized for clarity, ease of use, and execution:

```
Arabic-NLP-Project/
├── data/
│   ├── ANERCorp.conll        # Arabic NER Data (Sample from ANERCorp)
│   └── ar_padt-ud-train.conllu # Arabic POS Data (UD Arabic PADT)
├── NER_models/
│   ├── hmm_ner.py            # Hidden Markov Model for NER
│   ├── crf_ner.py            # Conditional Random Field for NER
│   └── bert_ner.py           # BERT-based model for NER (using AraBERT)
├── POS_models/
│   ├── rule_based_pos.py     # Rule-based model for POS
│   ├── maxent_pos.py         # Maximum Entropy model for POS
│   └── bilstm_pos.py         # BiLSTM model for POS
├── scripts/
│   ├── data_utils.py         # Helper functions for loading and processing data
│   ├── train_ner_models.py   # Script to train and evaluate all NER models
│   └── train_pos_models.py   # Script to train and evaluate all POS models
├── outputs/
│   └── *.txt                 # Evaluation reports generated after training
├── requirements.txt          # List of all required Python libraries
└── README.md                 # This file
```

Implemented Models

| Task | Model | Approach | File |
| :--- | :--- | :--- | :--- |
| NER | Hidden Markov Model (HMM) | Classical Statistical | `NER_models/hmm_ner.py` |
| NER | Conditional Random Field (CRF) | Classical ML (Feature-based) | `NER_models/crf_ner.py` |
| NER | BERT-based | Deep Learning (Fine-tuning AraBERT) | `NER_models/bert_ner.py` |
| POS | Rule-based | Rule-based (Simple Baseline) | `POS_models/rule_based_pos.py` |
| POS | Maximum Entropy (MaxEnt) | Classical ML (Logistic Regression) | `POS_models/maxent_pos.py` |
| POS | BiLSTM | Deep Learning (Bidirectional Recurrent Neural Network) | `POS_models/bilstm_pos.py` |

How to Run the Project

1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

2. Install Dependencies

Install all required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

3. Data Preparation

The necessary Arabic datasets are included in the `data/` folder:
*   `data/ANERCorp.conll` (for NER)
*   `data/ar_padt-ud-train.conllu` (for POS)

Note: The `scripts/data_utils.py` script is configured to read these local files.

4. Training and Evaluation

You can run the training and evaluation scripts for each task separately:

Train and Evaluate NER Models

```bash
python scripts/train_ner_models.py
```

This script will:
1.  Load the NER data.
2.  Split the data into training and testing sets.
3.  Train and evaluate the HMM, CRF, and BERT models.
4.  Save the evaluation reports in the `outputs/` folder.

Train and Evaluate POS Models

```bash
python scripts/train_pos_models.py
```

This script will:
1.  Load the POS data.
2.  Split the data into training and testing sets.
3.  Train and evaluate the Rule-based, MaxEnt, and BiLSTM models.
4.  Save the evaluation reports in the `outputs/` folder.

Running on Google Colab or VS Code

Google Colab (Recommended for Deep Learning Models)

1.  Clone the Repository: Open a new Colab notebook and run the following commands to clone the project:
    ```python
    !git clone https://github.com/KerimOdeh951/NER-POS-NLP-Project.git
    %cd NER-POS-NLP-Project
    ```
2.  Install Dependencies:
    ```python
    !pip install -r requirements.txt
    ```
3.  Run Training: Execute the training scripts using `!python`.

VS Code / PyCharm

1.  Open Folder: Open the `Arabic-NLP-Project` folder in your IDE.
2.  Virtual Environment: (Recommended) Create and activate a virtual environment.
3.  Install Dependencies: Run `pip install -r requirements.txt`.
4.  Run Training: Execute the training scripts from your IDE's terminal.

Important Notes

*   Deep Learning Models (BERT & BiLSTM): Training these models is resource-intensive. It is highly recommended to use Google Colab or Kaggle Notebooks to leverage their free GPU resources for faster training. The provided scripts use a small subset of the data for these models to ensure quick execution.
*   Data: The project uses a sample of the widely-used `ANERCorp` for NER and the `UD_Arabic-PADT` corpus for POS, which are standard Arabic NLP datasets.
*   Evaluation: Standard metrics (`seqeval` for NER and `classification_report` for POS) are used to ensure accurate token-level evaluation.
