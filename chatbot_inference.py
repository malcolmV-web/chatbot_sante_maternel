#!/usr/bin/env python3
"""
chatbot_inference.py

Charge les artefacts d'entraînement et expose :
 - `predict(text)` -> (answer, score, method)
 - CLI interactif (Question > ...)
    - Snippet Flask pour intégration facile dans une API web

Usage (CLI):
    python chabot_inference.py --tfidf tfidf.pkl --tokenizer tokenizer.pkl --label label_encoder.pkl --model chatbot_model.h5 --data data_clean.csv

Notes:
 - Pour le fallback TF-IDF, fournis le CSV nettoyé (data_clean.csv) contenant colonnes:
     question_raw, answer_raw, question (lemmatisée)
 - spaCy model (par défaut fr_core_news_sm) est utilisé pour la normalisation/lemmatisation.
"""
import os
import re
import sys
import argparse
import logging
import pickle
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import spacy
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Helpers : normalization / lemmatisation (identique au preprocessing)
# ---------------------------
_non_word_re = re.compile(r"[^a-z0-9'\-\s]")

def basic_normalize(text: str) -> str:
    """Nettoyage de surface : suppression de $, \\, /, mot 'text', accents, keep letters/numbers/apostrophe/hyphen"""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = s.replace("$", " ").replace("\\", " ").replace("/", " ")
    s = re.sub(r"\btext\b", " ", s)
    # Normaliser accents
    import unicodedata
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    s = _non_word_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def spacy_lemmatize(nlp, text: str, min_token_len: int = 2, remove_stopwords: bool = True) -> str:
    if not text:
        return ""
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if remove_stopwords and tok.is_stop:
            continue
        lem = tok.lemma_.lower().strip() or tok.text.lower()
        if re.fullmatch(r"[a-z0-9'\-]+", lem) and len(lem) >= min_token_len:
            tokens.append(lem)
    return " ".join(tokens)

# ---------------------------
# Core class that loads resources and predicts
# ---------------------------
class ChatbotInference:
    def __init__(
        self,
        tfidf_path: str,
        tokenizer_path: str,
        label_encoder_path: str,
        model_path: str,
        data_csv: Optional[str] = None,
        spacy_model: str = "fr_core_news_sm",
    ):
        # load spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"spaCy model '{spacy_model}' chargé.")
        except Exception as e:
            logger.error(f"Impossible de charger spaCy model '{spacy_model}': {e}")
            raise

        # load TF-IDF (optional but recommended)
        self.tfidf = None
        self.questions = []
        self.answers = []
        self.tfidf_matrix = None

        if tfidf_path and os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                self.tfidf = pickle.load(f)
            logger.info(f"TF-IDF chargé depuis: {tfidf_path}")
        else:
            logger.warning("TF-IDF introuvable : le fallback sémantique ne sera pas disponible.")

        # load tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer introuvable: {tokenizer_path}")
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        logger.info(f"Tokenizer chargé depuis: {tokenizer_path}")

        # load label encoder
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder introuvable: {label_encoder_path}")
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        logger.info(f"Label encoder chargé depuis: {label_encoder_path}")

        # load keras model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        self.model = load_model(model_path)
        logger.info(f"Modèle Keras chargé depuis: {model_path}")

        # optionally load CSV cleaned to build TF-IDF matrix for semantic search
        if data_csv and os.path.exists(data_csv):
            df = pd.read_csv(data_csv, dtype=str)
            # prefer 'question' column (lemmatisée) if present, fallback to first column
            if 'question' in df.columns:
                self.questions = df['question'].fillna('').astype(str).tolist()
            elif 'question_lemm' in df.columns:
                self.questions = df['question_lemm'].fillna('').astype(str).tolist()
            else:
                # try first column
                self.questions = df.iloc[:, 0].fillna('').astype(str).tolist()
            # answers
            if 'answer_raw' in df.columns:
                self.answers = df['answer_raw'].fillna('').astype(str).tolist()
            elif df.shape[1] >= 2:
                self.answers = df.iloc[:, 1].fillna('').astype(str).tolist()
            else:
                self.answers = [''] * len(self.questions)

            # If tfidf loaded but matrix not persisted, build matrix from loaded tfidf on the stored questions.
            if self.tfidf is not None:
                try:
                    self.tfidf_matrix = self.tfidf.transform(self.questions)
                    logger.info(f"TF-IDF matrix bâtie à partir de {len(self.questions)} questions du CSV.")
                except Exception as e:
                    logger.warning("Impossible de transformer les questions avec TF-IDF chargé: %s", e)
                    self.tfidf_matrix = None
            else:
                # build new tfidf if none provided
                try:
                    self.tfidf = TfidfVectorizer()
                    self.tfidf_matrix = self.tfidf.fit_transform(self.questions)
                    logger.info("TF-IDF entraîné localement à partir du CSV (fallback).")
                except Exception as e:
                    logger.warning("Impossible d'entraîner TF-IDF à partir du CSV: %s", e)
                    self.tfidf = None
                    self.tfidf_matrix = None
        else:
            if data_csv:
                logger.warning(f"data_csv fourni mais introuvable: {data_csv}")
            # try to build tfidf_matrix = None -> fallback disabled
            self.questions = []
            self.answers = []
            self.tfidf_matrix = None

        # store tokenizer max_len: infer from tokenizer if possible, else use 50
        self.max_len = getattr(self.tokenizer, 'max_len', None) or None
        # it's safer to compute max_len from tokenizer.word_index? we will keep dynamic padding during predict
        logger.info("Inference ready.")

    # ---------------------------
    # semantic fallback using TF-IDF cosine similarity
    # ---------------------------
    def semantic_lookup(self, user_text: str, top_k: int = 5) -> list:
        """
        Retourne la liste des candidats triés par score: [{'idx','question','answer','score'}...]
        """
        out = []
        if not self.tfidf or self.tfidf_matrix is None or not self.questions:
            return out

        cleaned = basic_normalize(user_text)
        lemm = spacy_lemmatize(self.nlp, cleaned)
        vec = self.tfidf.transform([lemm])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        idxs = np.argsort(-sims)[:top_k]
        for i in idxs:
            out.append({"idx": int(i), "question": self.questions[i], "answer": self.answers[i] if i < len(self.answers) else "", "score": float(sims[i])})
        return out

    # ---------------------------
    # model-based prediction
    # ---------------------------
    def predict_model(self, user_text: str) -> Tuple[str, float]:
        """
        Prétraitement -> tokenizer -> pad -> model.predict -> label_encoder.inverse_transform
        Retourne (answer_text, confidence)
        """
        cleaned = basic_normalize(user_text)
        lemm = spacy_lemmatize(self.nlp, cleaned)
        seq = self.tokenizer.texts_to_sequences([lemm])
        # pad sequences: infer max_len from model input shape if possible
        input_shape = getattr(self.model.input_shape, "__len__", None) and self.model.input_shape
        if input_shape and isinstance(input_shape, tuple):
            # model.input_shape may be (None, max_len)
            try:
                model_max_len = int(input_shape[1])
            except Exception:
                model_max_len = max(10, max(len(s) for s in seq))
        else:
            # fallback
            model_max_len = max(10, max(len(s) for s in seq))
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq_padded = pad_sequences(seq, maxlen=model_max_len, padding="post")
        preds = self.model.predict(seq_padded)
        prob = float(np.max(preds))
        pred_idx = int(np.argmax(preds, axis=1)[0])
        try:
            answer = self.label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            # If label encoder stores classes in different format (strings), handle gracefully
            classes = getattr(self.label_encoder, "classes_", None)
            if classes is not None and pred_idx < len(classes):
                answer = classes[pred_idx]
            else:
                answer = ""
        return answer, prob

    # ---------------------------
    # predict wrapper: tries semantic fallback first, then model, returns dict
    # ---------------------------
    def predict(self, user_text: str, semantic_threshold: float = 0.70, model_conf_threshold: float = 0.45) -> Dict:
        """
        Retourne dictionnaire:
          {
            "answer": str,
            "score": float,
            "method": "tfidf" | "model" | "low_conf" | "no_match",
            "candidates": [ ... ]  # optional list when tfidf used
          }
        """
        # 1) TF-IDF semantic lookup (if available)
        if self.tfidf is not None and self.tfidf_matrix is not None:
            candidates = self.semantic_lookup(user_text, top_k=5)
            if candidates:
                best = candidates[0]
                if best["score"] >= semantic_threshold:
                    return {"answer": best["answer"], "score": best["score"], "method": "tfidf", "candidates": candidates}
            # keep candidates to return later even if below threshold
        else:
            candidates = []

        # 2) model prediction
        answer, prob = self.predict_model(user_text)
        if prob >= model_conf_threshold and answer:
            return {"answer": answer, "score": prob, "method": "model", "candidates": candidates}
        # 3) low confidence
        # if semantic candidates exist we may still return top candidate even if below threshold (optional)
        if candidates:
            # return best candidate even if below semantic_threshold (less preferred)
            return {"answer": candidates[0]["answer"], "score": candidates[0]["score"], "method": "tfidf_low", "candidates": candidates}

        return {"answer": "Je ne suis pas sûr de comprendre. Peux-tu reformuler ou préciser ?", "score": float(prob), "method": "low_conf", "candidates": candidates}

# ---------------------------
# CLI & Flask helper
# ---------------------------
def interactive_loop(bot: ChatbotInference):
    print("Mode interactif — tape 'exit' pour quitter.")
    try:
        while True:
            q = input("Question > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            res = bot.predict(q)
            print(f"\n=> Méthode: {res['method']} | score: {res['score']:.3f}\nRéponse: {res['answer']}\n")
            if res.get("candidates"):
                print("Top candidates (tf-idf):")
                for c in res["candidates"]:
                    print(f"  - score={c['score']:.3f} q={c['question']}")
                print()
    except KeyboardInterrupt:
        print("\nInterrompu. Bye.")

# Example minimal Flask integration (to paste in your app.py)
FLASK_SNIPPET = """
# from flask import Flask, request, jsonify
# from inference import ChatbotInference
#
# bot = ChatbotInference(tfidf_path='tfidf.pkl', tokenizer_path='tokenizer.pkl',
#                        label_encoder_path='label_encoder.pkl', model_path='chatbot_model.h5',
#                        data_csv='data/data_clean.csv')
#
# app = Flask(__name__)
#
# @app.route('/chat', methods=['POST'])
# def chat():
#     payload = request.json or {}
#     text = payload.get('message', '')
#     if not text:
#         return jsonify({'error':'empty message'}), 400
#     res = bot.predict(text)
#     return jsonify({'message': res['answer'], 'score': res['score'], 'method': res['method']})
#
# if __name__ == '__main__':
#     app.run(debug=True)
"""

# ---------------------------
# main: argument parsing
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfidf", default="tfidf.pkl", help="Chemin vers tfidf.pkl (optionnel)")
    p.add_argument("--tokenizer", default="tokenizer.pkl", help="Chemin vers tokenizer.pkl")
    p.add_argument("--label", default="label_encoder.pkl", help="Chemin vers label_encoder.pkl")
    p.add_argument("--model", default="chatbot_model.h5", help="Chemin vers le modèle Keras .h5")
    p.add_argument("--data", default=None, help="CSV nettoyé (data_clean.csv) pour TF-IDF fallback (optionnel)")
    p.add_argument("--spacy", default="fr_core_news_sm", help="Nom du modèle spaCy")
    args = p.parse_args()

    bot = ChatbotInference(tfidf_path=args.tfidf, tokenizer_path=args.tokenizer, label_encoder_path=args.label, model_path=args.model, data_csv=args.data, spacy_model=args.spacy)
    print("Inference ready. Pour intégrer dans Flask, voir le snippet dans le fichier.")
    interactive_loop(bot)

if __name__ == "__main__":
    main()
