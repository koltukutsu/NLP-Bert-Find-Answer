import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch




# Define function to get answer and highlight it in text
def get_answer(question, text):
    # Load BERT-QA model
    tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/bert-multi-cased-finetuned-xquadv1"
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        "mrm8488/bert-multi-cased-finetuned-xquadv1"
    )

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
        answer, f"<mark style='background-color: yellow'>{answer}</mark>"
    )
    print(answer)
    # highlighted_text = text
    del model
    del inputs
    del tokenizer
    del input_ids
    del attention_mask
    del start_index
    del end_index

    return highlighted_text

logo = Image.open("ytu.png")
st.image(logo, width=200)

# Define Streamlit app
st.title("BERT Dil Modeli ile Soru Cevaplandırma")
question = st.text_input("Soru")
text = st.text_area("Sorunun Cevabının Aranacağı Metin")

if st.button("Cevap Al"):
    if not question:
        st.warning("Lütfen Soruyu Girin.")
    elif not text:
        st.warning("Lütfen Sorunun Aranacağı Metni Girin.")
    else:
        highlighted_text = get_answer(question, text)
        st.markdown(highlighted_text, unsafe_allow_html=True)
