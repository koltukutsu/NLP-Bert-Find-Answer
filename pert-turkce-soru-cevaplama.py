import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Define function to get answer and highlight it in text
def get_answer(question, text, tokenizer, model):
    # Tokenize inputs
    inputs = tokenizer.encode_plus(
        question, text, add_special_tokens=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Get start and end logits
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # Find answer
    start_index = torch.argmax(start_logits, dim=1)[0]
    end_index = torch.argmax(end_logits, dim=1)[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[0][start_index : end_index + 1])
    )

    # Highlight answer in text
    highlighted_text = text.replace(
        answer, f"<mark style='background-color: #b5e48c'>{answer}</mark>"
    )

    return (answer, highlighted_text)

logo = Image.open("ytu.png")
st.image(logo, width=200)

# Define Streamlit app
st.title("BERT Dil Modeli ile Soru Cevaplandırma")
st.subheader("Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği Bölümü Ara Projesi")
st.subheader("Erdi YAĞCI - Ç18061067")
st.subheader("Mehmet Semih BABACAN - Ç18069040")

# Load BERT-QA model outside the function to avoid reloading it each time.
name_of_repo = "daddycik/bert-turkce-soru-cevaplama"
tokenizer = AutoTokenizer.from_pretrained(name_of_repo)
model = AutoModelForQuestionAnswering.from_pretrained(name_of_repo)

question = st.text_input("Soru")
text = st.text_area("Sorunun Cevabının Aranacağı Metin")
st.text("(Sorunun aranacığı metin 512 Token'dan uzun olmamalı.)")

# Tokenize the text to count the tokens
tokens = tokenizer.encode(text, truncation=False)
token_size = len(tokens)
if token_size > 512:
    st.warning(f"Sorunun aranacağı metin 512 token size'dan daha uzun. Metnin uzunluğu {token_size} tokendır. Lütfen metni kısaltın.")
else:
    if st.button("Cevap Al"):
        if not question:
            st.warning("Lütfen Soruyu Girin.")
        elif not text:
            st.warning("Lütfen Sorunun Aranacağı Metni Girin.")
        else:
            answer, highlighted_text = get_answer(question, text, tokenizer, model)
            st.markdown(f'Cevap: "{answer}"', unsafe_allow_html=True)
            st.markdown(f"Cevap verilen metnin içinde:\n{highlighted_text}", unsafe_allow_html=True)
