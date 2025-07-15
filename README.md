# RoGuard: Advancing Safety for LLMs with Robust Guardrails


<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/Roblox/RoGuard" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoGuard-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://huggingface.co/datasets/Roblox/RoGuard-Eval" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoGuardEval-ffc107?color=1783ff&logoColor=white"/></a>
</div>

<p align="center">
<a href="https://devforum.roblox.com/t/beta-introducing-text-generation-api/3556520" target="_blank"><img src=https://img.shields.io/badge/Roblox-Blog-000000.svg?logo=Roblox height=22px></a>
<img src=https://img.shields.io/badge/ArXiv-Report-b5212f.svg?logo=arxiv height=22px> (coming soon)
</p>

RoGuard, a SOTA instruction fine-tuned LLM, is designed to help safeguard our Text Generation API. It performs safety classification at both the prompt and response levels, deciding whether or not each input or output violates our policies. This dual-level assessment is essential for moderating both user queries and the model’s own generated outputs. At the heart of our system is an LLM that’s been fine-tuned from the Llama-3.1-8B-Instruct model. We trained this LLM with a particular focus on high-quality instruction tuning to optimize for safety judgment performance.  


## 📦 Installation
Install the required dependencies:
```
python -m venv venv_roguard
source venv_roguard/bin/activate 
pip install -r requirements.txt
```

## 🧠 Inference
Run safety evaluations:
```
python inference.py --config configs/RoGuard.json
```

## ⚙️ Configuration
Multiple configuration files are already prepared and ready to use in the `configs/` folder.

To run an evaluation, each config file (in JSON format) should follow this structure:
```
{
  "name": "RoGuard",                                            // Eval dataset name

  "model_path": "Roblox/RoGuard",                               // Our model path in huggingface
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",        // Base model
  "max_output_tokens": 100,                                     // Max tokens the model can generate

  "eval_prompt": "prompts/RoGuard.txt",                         // Prompt template with {prompt}, {response}
  "llm_output_field": "Response Safety",                        // Key in model output to check prediction
  "llm_flagged_value": "unsafe",                                // Value representing "unsafe" in model output

  "eval_dataset": "Roblox/RoGuard-Eval",                        // Our eval dataset in huggingface
  "eval_label_field": "violation",                              // Field name indicating ground truth label for eval (optional for labeling-only runs)
  "eval_flagged_value": "true",                                 // Value representing "unsafe" in the dataset for eval (optional for labeling-only runs)

  "output_file": "outputs/RoGuard.csv"                          // Where to save CSV results
}
```


## 📄 Output Files

- **Evaluation Results** (`*.csv`):  
  - `input_prompt`: the original prompt  
  - `input_response`: the model’s generated response  
  - `actual_unsafe`: ground-truth label (if provided)  
  - `predicted_unsafe`: model’s prediction  
  - `correct`: whether the prediction matched the ground truth  

- **Summary Metrics** (`*_summary.csv`):  
  - Count-based metrics:  
    `Total Examples`, `True Positives`, `False Negatives`, `False Positives`, `True Negatives`  
  - Performance metrics (as percentages):  
    `Precision`, `Recall`, `F1 Score`, `False Positive Rate`


## 📁 Directory Structure
```
.
├── configs/                # Evaluation configs for different datasets
│   ├── aegis.json
│   ├── ...
│   └── RoGuard.json
├── prompts/                # Prompt files for inference or evaluation
│   ├── aegis.json
│   ├── ...
│   └── RoGuard.txt
├── outputs/                # Output CSVs for results and summaries
│   ├── RoGuard.csv
│   └── RoGuard_summary.csv
├── inference.py            # Script for running inference/evaluation
└── requirements.txt        # Python dependencies
```

## 📊 Model Benchmark Results

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
| RoGuard                   |   75.8  |  70.5 |  91.1 |   90.2 |   88.7 |     87.5 |      69.7 |   80.0 |   80.7 |
