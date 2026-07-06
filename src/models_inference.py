import os
import json
import numpy as np
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from src.preprocessing import clean_text, apply_keyword_enhancement

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BILSTM_DIR = os.path.join(BASE_DIR, "models", "bilstm_model")
BERT_DIR = os.path.join(BASE_DIR, "models", "bert_model")

class EmotionPredictor:
    def __init__(self):
        print("Loading BiLSTM assets...")
        self.bilstm_model = load_model(os.path.join(BILSTM_DIR, "bilstm_emotion_model.h5"))
        
        with open(os.path.join(BILSTM_DIR, "tokenizer_config.json"), "r") as f:
            tokenizer_data = json.load(f)
            if isinstance(tokenizer_data, str):
                self.bilstm_tokenizer = tokenizer_from_json(tokenizer_data)
            else:
                self.bilstm_tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
                
        self.classes = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']
        
        print("Loading BERT assets...")
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        self.bert_model = BertForSequenceClassification.from_pretrained(BERT_DIR)
        self.bert_model.eval()

    def predict_bilstm(self, text: str) -> dict:
        cleaned = clean_text(text)
        seq = self.bilstm_tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
        
        preds = self.bilstm_model.predict(padded, verbose=0)[0]
        prob_dict = {self.classes[i]: float(preds[i]) for i in range(len(self.classes))}
        return apply_keyword_enhancement(text, prob_dict)

    def predict_bert(self, text: str) -> dict:
        cleaned = clean_text(text)
        inputs = self.bert_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=50)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten().tolist()
            
        prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        return apply_keyword_enhancement(text, prob_dict)