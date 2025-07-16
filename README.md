# RoGuard: Advancing Safety for LLMs with Robust Guardrails


<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/Roblox/RoGuard" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoGuard-ffc107?color=ffc107&logoColor=white"/></a>
  
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/datasets/Roblox/RoGuard-Eval" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoGuardEval-ffc107?color=1783ff&logoColor=white"/></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC_BY_NC_SA_4.0-blue" alt="Data License"></a>
  <a href="https://github.com/Roblox/RoGuard/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code%20License-RAIL_MS-green" alt="Code License"></a>
</div>

<div align="center" style="line-height: 1;">
<a href="https://devforum.roblox.com/t/beta-introducing-text-generation-api/3556520" target="_blank"><img src=https://img.shields.io/badge/Roblox-Blog-000000.svg?logo=Roblox height=22px></a>
<img src="https://img.shields.io/badge/ArXiv-Report-b5212f.svg?logo=arxiv" height="22px"><sub>(coming soon)</sub>
</div>

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

<img width="1920" height="1338" alt="Metric" src="https://github.com/user-attachments/assets/847e3522-d579-4a1b-8a93-9c176470b1a4" />
