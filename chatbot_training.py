#!/usr/bin/env python3

"""
Prétraitement robuste + optionnel entraînement.

Usage examples:
    # Nettoyage uniquement
    python chatbot_training.py -i data/data.csv -o data/data_clean.csv

    # Nettoyage puis entraînement
    python chatbot_training.py -i data/data.csv -o data/data_clean.csv --train
"""
import os
import re
import unicodedata
import logging
import argparse
import pickle
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -------------------------
# Helpers: text normalization
# -------------------------
def normalize_text_basic(text: str) -> str:
    """Nettoyage de surface : suppression de $, \, /, mot 'text', accents, keep letters/numbers/apostrophe/hyphen"""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # Remplacer caractères indésirables
    s = s.replace("$", " ").replace("\\", " ").replace("/", " ")
    # supprimer le mot 'text' isolé (cas csv export)
    s = re.sub(r"\btext\b", " ", s)
    # normaliser accents
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')
    # ne garder que lettres, chiffres, apostrophes, traits d'union et espaces
    s = re.sub(r"[^a-z0-9'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_spacy(model_name: str = "fr_core_news_sm"):
    try:
        nlp = spacy.load(model_name)
        logging.info("spaCy model '%s' chargé.", model_name)
        return nlp
    except OSError:
        logging.error("Modèle spaCy %s introuvable. Lancez: python -m spacy download %s", model_name, model_name)
        raise


def spacy_lemmatize_and_filter(nlp, text: str, min_token_len: int = 2, remove_stopwords: bool = True) -> str:
    """Lemmatisation via spaCy ; filtre ponctuation/stopwords et tokens courts"""
    if not text:
        return ""
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        # prendre lemma si disponible sinon text
        lem = (tok.lemma_ or tok.text).lower().strip()
        if remove_stopwords and tok.is_stop:
            continue
        # autoriser a-z0-9 ' - et longueur minimale
        if re.fullmatch(r"[a-z0-9'\-]+", lem) and len(lem) >= min_token_len:
            tokens.append(lem)
    return " ".join(tokens)


# -------------------------
# CSV reading + column detection
# -------------------------
def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}

    q_candidates = ['question', 'questions', 'q', 'prompt', 'texte', 'text', 'question_raw', 'question_raw']
    a_candidates = ['reponse', 'réponse', 'reponses', 'answer', 'response', 'a', 'answer_raw', 'answer_raw']

    q_col = None
    a_col = None

    for cand in q_candidates:
        if cand in lower_map:
            q_col = lower_map[cand]
            break

    for cand in a_candidates:
        if cand in lower_map:
            a_col = lower_map[cand]
            break

    # fallback: first two columns
    if q_col is None or a_col is None:
        if len(cols) >= 2:
            if q_col is None:
                q_col = cols[0]
                logging.warning("Aucune colonne 'question' reconnue, j'utilise la première colonne: '%s' comme question.", q_col)
            if a_col is None:
                a_col = cols[1]
                logging.warning("Aucune colonne 'answer' reconnue, j'utilise la seconde colonne: '%s' comme réponse.", a_col)
        else:
            logging.error("Impossible de détecter colonnes question/réponse et moins de 2 colonnes présentes.")
            raise SystemExit(1)

    logging.info("Utilisation des colonnes -> question: '%s' | answer: '%s'", q_col, a_col)
    return q_col, a_col


# -------------------------
# Training utilities
# -------------------------
def build_model(vocab_size: int, input_length: int, n_classes: int, embed_dim: int = 128) -> keras.Model:
    inputs = layers.Input(shape=(input_length,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length)(inputs)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------
# Main processing function
# -------------------------
def preprocess_and_train(input_csv: str,
                         output_csv: str,
                         spacy_model: str = "fr_core_news_sm",
                         do_train: bool = False,
                         tfidf_path: str = "tfidf.pkl",
                         tokenizer_path: str = "tokenizer.pkl",
                         model_path: str = "chatbot_model.h5",
                         label_encoder_path: str = "label_encoder.pkl"):
    # 1) load spaCy
    nlp = load_spacy(spacy_model)

    # 2) read csv with flexible separator
    if not os.path.exists(input_csv):
        logging.error("Fichier introuvable: %s", input_csv)
        raise SystemExit(1)

    logging.info("Lecture du fichier : %s", input_csv)
    df = pd.read_csv(input_csv, sep=None, engine="python", dtype=str)
    logging.info("Colonnes trouvées: %s", list(df.columns))
    logging.info("Aperçu (head):\n%s", df.head(3).to_string(index=False))

    # 3) detect columns
    q_col, a_col = detect_columns(df)

    # 4) drop rows where either column is null/empty
    df[q_col] = df[q_col].astype(str)
    df[a_col] = df[a_col].astype(str)
    before = len(df)
    mask = df[q_col].str.strip().astype(bool) & df[a_col].str.strip().astype(bool)
    df = df[mask].copy()
    logging.info("Lignes avec champs vides supprimées: %d (reste %d)", before - len(df), len(df))
    if len(df) == 0:
        logging.error("Aucune donnée valide après suppression des lignes vides.")
        raise SystemExit(1)

    # 5) basic normalization
    logging.info("Application normalisation basique...")
    df['question_basic'] = df[q_col].apply(normalize_text_basic)
    df = df[df['question_basic'].str.strip().astype(bool)].copy()
    logging.info("Lignes après normalisation basique: %d", len(df))
    if len(df) == 0:
        logging.error("Aucune question après normalisation.")
        raise SystemExit(1)

    # 6) spaCy lemmatization
    logging.info("Lemmatisation spaCy (cela peut prendre un peu de temps)...")
    df['question_lemm'] = df['question_basic'].apply(lambda t: spacy_lemmatize_and_filter(nlp, t))
    before = len(df)
    df = df[df['question_lemm'].str.strip().astype(bool)].copy()
    logging.info("Lignes supprimées après lemmatisation vide: %d (reste %d)", before - len(df), len(df))
    if len(df) == 0:
        logging.error("Aucune donnée après lemmatisation.")
        raise SystemExit(1)

    # 7) drop duplicates (on lemmatized question)
    before = len(df)
    df.drop_duplicates(subset=['question_lemm'], inplace=True)
    logging.info("Doublons supprimés (question lemm): %d (reste %d)", before - len(df), len(df))

    # 8) prepare cleaned dataframe and save
    df_clean = df[[q_col, a_col, 'question_lemm']].copy()
    df_clean.rename(columns={q_col: 'question_raw', a_col: 'answer_raw', 'question_lemm': 'question'}, inplace=True)

    outdir = os.path.dirname(output_csv) or "."
    os.makedirs(outdir, exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    logging.info("CSV nettoyé sauvegardé dans: %s (%d lignes)", output_csv, len(df_clean))

    if not do_train:
        logging.info("Mode nettoyage uniquement; pour entraîner, relancer avec --train")
        return

    # ---- Training ----
    logging.info("Démarrage entraînement...")

    questions = df_clean['question'].tolist()
    answers = df_clean['answer_raw'].tolist()

    # TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(questions)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    logging.info("TF-IDF sauvegardé: %s", tfidf_path)

    # label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(answers)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logging.info("Label encoder sauvegardé: %s", label_encoder_path)

    # tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(questions)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logging.info("Tokenizer sauvegardé: %s", tokenizer_path)

    # sequences
    sequences = tokenizer.texts_to_sequences(questions)
    max_len = max(10, max(len(s) for s in sequences))
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post")
    vocab_size = len(tokenizer.word_index) + 1
    n_classes = len(label_encoder.classes_)
    logging.info("Vocab size %d | max_len %d | classes %d", vocab_size, max_len, n_classes)

    # build & train model
    model = build_model(vocab_size=vocab_size, input_length=max_len, n_classes=n_classes, embed_dim=128)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]
    history = model.fit(sequences, y, epochs=30, batch_size=16, validation_split=0.15, callbacks=callbacks, verbose=2)
    model.save(model_path)
    logging.info("Modèle sauvegardé: %s", model_path)

    # evaluation (sur training set)
    y_pred = np.argmax(model.predict(sequences), axis=1)
    from sklearn.metrics import classification_report, accuracy_score
    report = classification_report(y, y_pred, zero_division=0)
    acc = accuracy_score(y, y_pred)
    logging.info("Classification report (train):\n%s", report)
    logging.info("Accuracy (train): %.4f", acc)
    logging.info("Entraînement terminé.")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prétraitement + (optionnel) entraînement du chatbot (spaCy lemmatisation).")
    p.add_argument("--input", "-i", default="/home/malcolmv/Documents/UV-BF_Genie logiciel/L3/S6/projet_tutores/chatbot_sante_maternel/data/data.csv", help="CSV input")
    p.add_argument("--output", "-o", default="/home/malcolmv/Documents/UV-BF_Genie logiciel/L3/S6/projet_tutores/chatbot_sante_maternel/data/data_clean.csv", help="CSV nettoyé output")
    p.add_argument("--spacy_model", default="fr_core_news_sm", help="Nom du modèle spaCy")
    p.add_argument("--train", action="store_true", help="Lance entraînement après nettoyage")
    p.add_argument("--tfidf_path", default="tfidf.pkl", help="Chemin TF-IDF pickle")
    p.add_argument("--tokenizer_path", default="tokenizer.pkl", help="Chemin tokenizer pickle")
    p.add_argument("--model_path", default="chatbot_model.h5", help="Chemin pour sauvegarder le modèle keras")
    p.add_argument("--label_encoder_path", default="label_encoder.pkl", help="Chemin label encoder pickle")
    return p.parse_args()


def main():
    args = parse_args()
    preprocess_and_train(
        input_csv=args.input,
        output_csv=args.output,
        spacy_model=args.spacy_model,
        do_train=args.train,
        tfidf_path=args.tfidf_path,
        tokenizer_path=args.tokenizer_path,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path
    )


if __name__ == "__main__":
    main()