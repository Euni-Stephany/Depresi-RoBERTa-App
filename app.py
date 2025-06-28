import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

model_id = "Eunii/Depresi-RoBERTa-App"

# # Load dari model lokal
# model_path = "content/model_mental_roberta"
tokenizer = RobertaTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(model_id)
model.eval()

# Cek isi bobot model
print("🔍 Bobot model dimuat? →", list(model.state_dict().keys())[:5])

# Custom predict function
# Prediction function with confidence score
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
    return predicted_class, confidence

# Streamlit App Configuration
st.set_page_config(page_title="🧠 Mental Health Detection", layout="wide")
st.title("🧠 Mental Health Detection Based on Expressive Sentence")
st.markdown("Enter an expressive sentence (e.g., like from a tweet) to detect whether it indicates signs of depression.")

input_text = st.text_area("📝 Enter your sentence:", height=200, placeholder="Example: I feel so tired of life lately...")

if st.button("🔍 Detect Now"):
    if input_text.strip():
        label, confidence = predict(input_text)

        if label == 1:
            st.markdown(
                f"""
                <div style='padding:20px; border-radius:10px; background-color:#525452'>
                    <h2 style='color:#cc0000;'>🟥 Result: Depression Detected</h2>
                    <p>This sentence shows signs of possible depression. Please consider reaching out to a mental health professional, friend, or trusted person.</p>
                    <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='padding:20px; border-radius:10px; background-color:#525452dart'>
                    <h2 style='color:#0066cc;'>🟩 Result: No Depression Detected</h2>
                    <p>This sentence does not show significant signs of depression based on model analysis.</p>
                    <p><b>Confidence:</b> {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter a sentence first.")