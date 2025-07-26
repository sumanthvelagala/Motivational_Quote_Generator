Fine-tuned the TinyLlama-1.1B-Chat model on a custom dataset of motivational quotes labeled by topic. Created an interactive UI that takes a user’s topic and generates motivational quotes from both the base model and the fine-tuned model, allowing side-by-side comparison. The fine-tuned model produces more original and relevant quotes, helping users see the improvement clearly.


# Project File Structure 
motivation-llm/
├── quotes.csv                     # Dataset
├── train.py                       # LoRA fine-tuning script
├── front.py                       # UI to try out the fine-tuned model
├── Fine_Tuned_5000                # Saved fine_tuned model 
└── README.md

# Training Pipeline Summary
- Randomly selected 5,000 samples from the full Kaggle Quotes Dataset for efficient experimentation.
- Transformed each sample into instruction-style prompt-response pairs, such as:
    - Prompt: Write a motivational quote about {topic}
    - Response: "Success doesn't come overnight..."
- Applied manual padding and created attention_mask and labels aligned with instruction-tuned model requirements.
- Wrapped the data in a custom PyTorch Dataset class to return preprocessed tensors.
- Used DataLoader for efficient batching.
- Fine-tuned the model using LoRA (Low-Rank Adaptation) on top of TinyLlama/TinyLlama-1.1B-Chat-v1.0 with the following settings:
    -Optimizer: AdamW
    - Learning Rate Scheduling: With warmup steps of 1% of total steps
    - Early Stopping: Triggered if no improvement after 2 consecutive epochs
    - Max Epochs: 8


Base-Model - TinyLlama/TinyLlama-1.1B-Chat-v1.0
DataSet - https://www.kaggle.com/datasets/manann/quotes-500k
LoRA-Config - rank 8, alpha 16, applied on q_proj, v_proj


To-Train:
- Change the path to quotes.csv dataset path
- Change the path where the model weights to be saved


To-Run:
-Change the path of model and tokenizer with fine-tuned model weights and then run below command on terminal
-Streamlit run front.py



