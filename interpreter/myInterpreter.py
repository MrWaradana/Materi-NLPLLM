from flask import Flask, request, jsonify
from transformers import pipeline
import re

app = Flask(__name__)

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def hybrid_entity_recognition(text):
    # 1. BERT-based NER
    bert_entities = ner_pipeline(text)

    # Standardisasi key dari BERT ke format seragam
    standardized_bert_entities = [
        {
            "entity": ent["entity_group"],
            "word": ent["word"],
            "score": ent["score"],
            "start": ent["start"],
            "end": ent["end"],
        }
        for ent in bert_entities
    ]

    # 2. Regex-based NER
    regex_patterns = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}",
        "DATE": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "URL": r"https?://[^\s]+",
        "NRP": r"\b52\d+\b"
        }

    regex_entities = []
    for label, pattern in regex_patterns.items():
        for match in re.finditer(pattern, text):
            regex_entities.append({
                "entity": label,
                "word": match.group(),
                "score": 1.0,   # Regex tidak pakai probabilitas
                "start": match.start(),
                "end": match.end(),
            })

    # 3. Gabungkan hasil BERT + Regex
    all_entities = standardized_bert_entities + regex_entities

    return all_entities


#============================================

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

def load_intent_model(model_path="my_intent_model"):
    """Load model, tokenizer, dan label encoder."""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    model.eval()
    return model, tokenizer, label_encoder

def get_intent(text, model, tokenizer, label_encoder, entity_list, modtext):
    """Prediksi intent dari input text."""
    inputs = tokenizer(modtext, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        intent_name = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()

    if (text=='/session_start'):
        intent_name = 'session_start'
        confidence = '1.0'
       


    return {
        "intent": {
            "name": intent_name,
            "confidence": float(confidence)
        },
        "entities": entity_list,
        "text": text,
        "modtext": modtext
    }

model, tokenizer, label_encoder = load_intent_model("my_intent_model")

def replace_entity(text, entity_value, placeholder):
    # Escape entity_value biar aman kalau ada karakter khusus (seperti . atau @)
    pattern = re.escape(entity_value)
    return re.sub(pattern, placeholder, text)


def get_intent_and_entity(text):
    entities = hybrid_entity_recognition(text)
    entity_list = []
    modtext = text
    for ent in entities:
        entity_type = ent['entity']
        entity_typex = entity_type
        
        # value dari entity_typex harus sama dengan slot pada domain.yml
        if (entity_type=='PER'):
            entity_typex = 'name'
        if (entity_type=='NRP'):
            entity_typex = 'nrp'
        
        my_entity = {'value': ent['word'], 'entity': entity_typex}
        entity_list.append(my_entity)
        
        # modtext adalah user_input yang digunakan untuk prediksi intent 
        if (entity_type=='NRP'):
            modtext = replace_entity(text, ent['word'], "<nrp>")
        if (entity_type=='EMAIL'):
            modtext = replace_entity(text, ent['word'], "<email>")
        if (entity_type=='PER'):
            modtext = replace_entity(text, ent['word'], "<name>")
            
        #print(f"{ent['word']:<30} --> {ent['entity']} (start={ent['start']}, end={ent['end']})")
        
    hasil = get_intent(text, model, tokenizer, label_encoder, entity_list, modtext)

    return hasil 


@app.route("/model/parse", methods=["POST"])
def parse():
    data = request.get_json()
    user_input = data.get("text", "")
    
    print(f"Text: {user_input}")

    # Return RASA-style response
    response = get_intent_and_entity(user_input)

    print (response)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)


