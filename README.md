# 🛡️ RoGuard: Advancing Safety for LLMs with Robust Guardrails
RoGuard is a lightweight, modular evaluation framework for assessing the safety of fine-tuned language models. It provides structured evaluation using configurable prompts, labeled datasets, and outputs comprehensive metrics.

# 📦 Installation
Install the required dependencies:
```
python -m venv venv_roguard
source venv_roguard/bin/activate 
pip install -r requirements.txt
```

# ⚙️ Configuration
Prepare a config file (JSON format) with the following keys:
```
{
  "name": "roblox",                                             // Eval dataset name

  "model_path": "Roblox/RoGuard",                               // Our model path in huggingface
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",        // Base model
  "max_output_tokens": 100,                                     // Max tokens the model can generate

  "eval_prompt": "prompts/roblox.txt",                          // Prompt template with {prompt}, {response}
  "llm_output_field": "Response Safety",                        // Key in model output to check prediction
  "llm_flagged_value": "unsafe",                                // Value representing "unsafe" in model output

  "eval_dataset": "Roblox/RoGuard-Eval",                        // Our eval dataset in huggingface
  "eval_label_field": "violation",                              // Field name indicating ground truth label for eval (optional for labeling-only runs)
  "eval_flagged_value": "true",                                 // Value representing "unsafe" in the dataset for eval (optional for labeling-only runs)

  "output_file": "outputs/roblox.csv"                           // Where to save CSV results
}
```

# 🚀 Running Evaluation
To evaluate model safety on a labeled dataset:
```
python inference.py --config configs/roblox.json
```

# 🏷️ Running Labeling-Only (No Ground Truth)
You can run RoGuard on unlabeled data to generate model predictions only:
```
python inference.py --config configs/YOUR_LABELING_CONFIG.json
```

# 📄 Output Files
- Evaluation Results (CSV): Includes prompt, response, prediction, and correctness (if labels are provided).
- Summary Metrics (CSV): Available for evaluation runs only.
  - Includes: Precision, Recall, F1 Score, False Positive Rate
  - Confusion Matrix: TP / FP / TN / FN


# 📁 Directory Structure
```
.
├── configs/
│   └── YOUR_EVAL_CONFIG.json
├── datasets/
│   └── YOUR_EVAL_DATASET.json
├── prompts/
│   └── YOUR_PROMPT.txt
├── outputs/
│   ├── YOUR_EVAL_OUTPUT.csv
│   └── YOUR_EVAL_OUTPUT_summary.csv
├── inference.py
└── requirements.txt
```

# 📊 Model Benchmark Results

- **Prompt Metrics**: These evaluate how well the model classifies or responds to potentially harmful **user inputs**
- **Response Metrics**: These measure how well the model handles or generates **responses**, ensuring its outputs are safe and aligned.


| Model / Metric            | Prompt  |       |       |        |        | Response |           |        |        |
|---------------------------|--------:|------:|------:|-------:|-------:|---------:|----------:|-------:|-------:|
|                           | ToxicC. | OAI   | Aegis | XSTest | WildP. | BeaverT. | SaferRLHF | WildR. | HarmB. |
| LlamaGuard2-8B            |   42.7  |  77.6 |  73.8 |   88.6 |   70.9 |     71.8 |      51.6 |   65.2 |   78.5 |
| LlamaGuard3-8B            |   50.9  |  79.4 |  74.8 |   88.3 |   70.1 |     69.7 |      53.7 |   70.2 |   84.9 |
| MD-Judge-7B               |     -   |    -  |    -  |     -  |     -  |     86.7 |      64.8 |   76.8 |   81.2 |
| WildGuard-7B              |   70.8  |  72.1 |  89.4 |   94.4 |   88.9 |     84.4 |      64.2 |   75.4 |   86.2 |
| ShieldGemma-7B            |   70.2  |  82.1 |  88.7 |   92.5 |   88.1 |     84.8 |      66.6 |   77.8 |   84.8 |
| GPT-4o                    |   68.1  |  70.4 |  83.2 |   90.2 |   87.9 |     83.8 |      67.9 |   73.1 |   83.5 |
| BingoGuard-phi3-3B        |   72.5  |  72.8 |  90.0 |   90.8 |   88.9 |     86.2 |      69.9 |   79.7 |   85.1 |
| BingoGuard-llama3.1-8B    |   75.7  |  77.9 |  90.4 |   94.9 |   88.9 |     86.4 |      68.7 |   80.1 |   86.4 |
| 🛡️ RoGuard                |   75.8  |  70.5 |  91.1 |   90.2 |   88.7 |     87.5 |      69.7 |   80.0 |   80.7 |


You can download the model from [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-RoGuard)](https://huggingface.co/Roblox/RoGuard)
