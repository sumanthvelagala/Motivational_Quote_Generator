import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


@st.cache_resource
def load_model():
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_weights_path = "/Users/Sumanth/Terminal/LLM/Fine_Tuned_5000"

    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path
    )
    base_model.eval()
    fine_tune = PeftModel.from_pretrained(base_model, lora_weights_path)
    fine_tune.eval()
    
    return tokenizer, base_model, fine_tune

tokenizer, base_model,fine_tune_model = load_model()


st.title("Motivational Quote Generator: Fine-Tuned vs Base Model")
user_topic = st.text_input("Enter a topic you'd like quote on:")

def gen_quote(model,tokenizer,prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()

if st.button("Generate Quote"):
    if user_topic.strip():
        with st.spinner("Generating..."):
            prompt = (
    f"Motivational quote example: "
    f"\"Success is not final, failure is not fatal: It is the courage to continue that counts.\"\n"
    f"Now write a motivational quote about {user_topic}:"
)
            base_model_output = gen_quote(base_model,tokenizer,prompt)
            fine_tuned_output = gen_quote(fine_tune_model,tokenizer,prompt)
            column1,column2 = st.columns(2)

            with column1:
                st.subheader("Base Model(Non-Fine-Tuned)")
                st.success(base_model_output)

            with column2:
                st.subheader("Fine-Tuned Model")
                st.success(fine_tuned_output)
    else:
        st.warning("Please enter a topic first.")


