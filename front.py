import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re


@st.cache_resource
def load_model():
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_weights_path = "/Users/Sumanth/Terminal/LLM/Fine_Tuned_5000"

    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()
    
    return tokenizer, model

tokenizer, model = load_model()


st.title("Motivational Quote Generator")
user_topic = st.text_input("Enter a topic you'd like motivation on:")

if st.button("Generate Quote"):
    if user_topic.strip():
        with st.spinner("Generating..."):
            prompt = (
    f"Motivational quote example: "
    f"\"Success is not final, failure is not fatal: It is the courage to continue that counts.\"\n"
    f"Now write a motivational quote about {user_topic}:"
)


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
            quote = decoded.replace(prompt, "").strip()

            st.session_state.generated_quote = quote
    else:
        st.warning("Please enter a topic first.")

if "generated_quote" in st.session_state:
    st.success(st.session_state.generated_quote)
