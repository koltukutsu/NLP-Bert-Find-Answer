import gradio as gr

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


# Define function to get answer and highlight it in text
def get_answer(question, text):
    # Load BERT-QA model
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
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
        "Semih", f"<mark style='background-color: yellow'>Semih</mark>"
    )

    highlighted_text = text
    # del model
    # del inputs
    # del tokenizer
    # del input_ids
    # del attention_mask
    # del start_index
    # del end_index

    return highlighted_text
    return "highlighted_text"


# Define Gradio app
inputs = [
    gr.inputs.Textbox("Soru"),
    gr.inputs.Textbox("Sorunun Cevabının Aranacağı Metin"),
]
outputs = gr.outputs.HTML(label="Cevap")

gr.Interface(
    get_answer, inputs, outputs, title="BERT Dil Modeli ile Soru Cevaplandırma"
).launch()
