from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================================================
# Configuration
# =========================================================
MODEL_NAME = "Blablablab/neurobiber"
CHUNK_SIZE = 512
SUBBATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 128

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "wildchat" / "en_wildchat.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "english_clause_candidates.parquet"

TARGET_ROLE = "assistant"
TARGET_LANGUAGE = "english"

# NeuroBiber feature inventory
BIBER_FEATURES = [
    "BIN_QUAN", "BIN_QUPR", "BIN_AMP", "BIN_PASS", "BIN_XX0", "BIN_JJ",
    "BIN_BEMA", "BIN_CAUS", "BIN_CONC", "BIN_COND", "BIN_CONJ", "BIN_CONT",
    "BIN_DPAR", "BIN_DWNT", "BIN_EX", "BIN_FPP1", "BIN_GER", "BIN_RB",
    "BIN_PIN", "BIN_INPR", "BIN_TO", "BIN_NEMD", "BIN_OSUB", "BIN_PASTP",
    "BIN_VBD", "BIN_PHC", "BIN_PIRE", "BIN_PLACE", "BIN_POMD", "BIN_PRMD",
    "BIN_WZPRES", "BIN_VPRT", "BIN_PRIV", "BIN_PIT", "BIN_PUBV", "BIN_SPP2",
    "BIN_SMP", "BIN_SERE", "BIN_STPR", "BIN_SUAV", "BIN_SYNE", "BIN_TPP3",
    "BIN_TIME", "BIN_NOMZ", "BIN_BYPA", "BIN_PRED", "BIN_TOBJ", "BIN_TSUB",
    "BIN_THVC", "BIN_NN", "BIN_DEMP", "BIN_DEMO", "BIN_WHQU", "BIN_EMPH",
    "BIN_HDG", "BIN_WZPAST", "BIN_THAC", "BIN_PEAS", "BIN_ANDC", "BIN_PRESP",
    "BIN_PROD", "BIN_SPAU", "BIN_SPIN", "BIN_THATD", "BIN_WHOBJ", "BIN_WHSUB",
    "BIN_WHCL", "BIN_ART", "BIN_AUXB", "BIN_CAP", "BIN_SCONJ", "BIN_CCONJ",
    "BIN_DET", "BIN_EMOJ", "BIN_EMOT", "BIN_EXCL", "BIN_HASH", "BIN_INF",
    "BIN_UH", "BIN_NUM", "BIN_LAUGH", "BIN_PRP", "BIN_PREP", "BIN_NNP",
    "BIN_QUES", "BIN_QUOT", "BIN_AT", "BIN_SBJP", "BIN_URL", "BIN_WH",
    "BIN_INDA", "BIN_ACCU", "BIN_PGAS", "BIN_CMADJ", "BIN_SPADJ", "BIN_X",
]

COMPLEMENT_FEATURES = {"BIN_THVC", "BIN_THAC", "BIN_TOBJ", "BIN_TSUB"}
CONTROL_FEATURES = {"BIN_WHCL", "BIN_WHOBJ", "BIN_WHSUB"}
VERB_CLASS_FEATURES = {"BIN_PRIV", "BIN_PUBV"}
OUTPUT_FEATURES = sorted(COMPLEMENT_FEATURES | CONTROL_FEATURES | VERB_CLASS_FEATURES)


# =========================================================
# Utilities
# =========================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_sentence_split(text: str) -> list[str]:
    """
    Lightweight sentence splitter.
    Good enough for Phase 2 screening.
    """
    if not isinstance(text, str):
        return []

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def normalize_conversation(conversation: Any) -> list[dict[str, Any]]:
    """
    Normalize WildChat conversation objects.
    Supports:
    - numpy.ndarray of dicts
    - list of dicts
    """
    if conversation is None:
        return []

    if isinstance(conversation, np.ndarray):
        conversation = conversation.tolist()

    if not isinstance(conversation, list):
        return []

    normalized = []
    for item in conversation:
        if isinstance(item, dict):
            normalized.append(item)

    return normalized


def extract_assistant_turns_english(
    conversation: Any,
    target_role: str = TARGET_ROLE,
    target_language: str = TARGET_LANGUAGE,
) -> list[tuple[int, str, dict[str, Any]]]:
    """
    Extract assistant turns in English only.

    Returns:
        list of (turn_index, content, turn_dict)
    """
    turns = normalize_conversation(conversation)
    results: list[tuple[int, str, dict[str, Any]]] = []

    for idx, turn in enumerate(turns):
        role = str(turn.get("role", "")).strip().lower()
        content = turn.get("content", "")
        language = str(turn.get("language", "")).strip().lower()

        if (
            role == target_role
            and language == target_language
            and isinstance(content, str)
            and content.strip()
        ):
            results.append((idx, content.strip(), turn))

    return results


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Chunk long texts by whitespace token count.
    """
    tokens = text.split()
    if not tokens:
        return []

    return [
        " ".join(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]


# =========================================================
# NeuroBiber loading and prediction
# =========================================================
def load_neurobiber(model_name: str = MODEL_NAME):
    device = get_device()

    # Use slow tokenizer to avoid fast-tokenizer parsing failure
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device

def predict_batch_features(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    chunk_size: int = CHUNK_SIZE,
    subbatch_size: int = SUBBATCH_SIZE,
) -> np.ndarray:
    """
    Predict NeuroBiber feature vectors for a batch of texts.
    Strategy:
    - chunk long texts
    - score each chunk
    - aggregate by max over chunks
    - threshold at 0.5
    """
    chunked_texts: list[str] = []
    text_chunk_ranges: list[tuple[int, int]] = []

    for text in texts:
        chunks = chunk_text(text, chunk_size=chunk_size)
        start = len(chunked_texts)
        chunked_texts.extend(chunks)
        end = len(chunked_texts)
        text_chunk_ranges.append((start, end))

    if not chunked_texts:
        return np.zeros((len(texts), len(BIBER_FEATURES)), dtype=int)

    chunk_probs_all = []

    for i in range(0, len(chunked_texts), subbatch_size):
        batch_chunks = chunked_texts[i:i + subbatch_size]

        encodings = tokenizer(
            batch_chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=chunk_size,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.sigmoid(outputs.logits)
            chunk_probs_all.append(probs.cpu())

    chunk_probs_tensor = torch.cat(chunk_probs_all, dim=0)

    predictions = []
    num_labels = chunk_probs_tensor.shape[1]

    for start, end in text_chunk_ranges:
        if start == end:
            pred = torch.zeros(num_labels)
        else:
            chunk_slice = chunk_probs_tensor[start:end]
            pred, _ = torch.max(chunk_slice, dim=0)
        predictions.append((pred > 0.5).int().numpy())

    return np.array(predictions)


# =========================================================
# Phase 2 preparation
# =========================================================
def flatten_english_assistant_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per English assistant sentence.
    Uses turn-level language, not only row-level language.
    """
    if "conversation" not in df.columns:
        raise ValueError("Input dataframe must contain a 'conversation' column.")

    rows: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        conversation_id = row.get("conversation_hash", row_idx)
        row_language = row.get("language")
        model = row.get("model")
        row_timestamp = row.get("timestamp")
        turn_count = row.get("turn")


        assistant_turns = extract_assistant_turns_english(
            row["conversation"],
            target_role=TARGET_ROLE,
            target_language=TARGET_LANGUAGE,
        )

        for assistant_turn_index, assistant_text, turn_meta in assistant_turns:
            assistant_turn_timestamp = turn_meta.get("timestamp", row_timestamp)
            assistant_turn_language = turn_meta.get("language", row_language)
            assistant_turn_id = turn_meta.get("turn_identifier")

            sentences = simple_sentence_split(assistant_text)

            for sentence_index, sentence in enumerate(sentences):
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "row_index": row_idx,
                        "model": model,
                        "row_language": row_language,
                        "turn_language": assistant_turn_language,
                        "row_timestamp": row_timestamp,
                        "assistant_turn_timestamp": assistant_turn_timestamp,
                        "turn_count": turn_count,
                        "assistant_turn_index": assistant_turn_index,
                        "assistant_turn_identifier": assistant_turn_id,
                        "sentence_index": sentence_index,
                        "sentence": sentence,
                    }
                )

    return pd.DataFrame(rows)


# =========================================================
# Phase 2 screening
# =========================================================
def run_clause_candidate_screening(
    sentence_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    prediction_batch_size: int = PREDICTION_BATCH_SIZE,
    keep_zero_feature_rows: bool = False,
) -> pd.DataFrame:
    """
    Run NeuroBiber over English assistant sentences and keep clause candidates.
    """
    if sentence_df.empty:
        return sentence_df.copy()

    batch_results = []

    for start in range(0, len(sentence_df), prediction_batch_size):
        batch_df = sentence_df.iloc[start:start + prediction_batch_size].copy()
        texts = batch_df["sentence"].tolist()

        preds = predict_batch_features(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        pred_df = pd.DataFrame(preds, columns=BIBER_FEATURES, index=batch_df.index)

        for feat in OUTPUT_FEATURES:
            batch_df[feat] = pred_df[feat].astype(int)

        batch_df["is_candidate"] = batch_df[list(COMPLEMENT_FEATURES)].max(axis=1).astype(int)

        if not keep_zero_feature_rows:
            batch_df = batch_df[batch_df["is_candidate"] == 1].copy()

        batch_results.append(batch_df)

    result_df = pd.concat(batch_results, ignore_index=True)

    return result_df.sort_values(
        by=["conversation_id", "assistant_turn_index", "sentence_index"]
    ).reset_index(drop=True)


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("Loading parquet dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded rows: {len(df)}")

    print("Inspecting top-level language distribution...")
    if "language" in df.columns:
        print(df["language"].value_counts(dropna=False).head(10))
    else:
        print("No top-level 'language' column found.")

    print("Flattening English assistant sentences...")
    sentence_df = flatten_english_assistant_sentences(df)
    print(f"English assistant sentences: {len(sentence_df)}")

    if sentence_df.empty:
        print("No English assistant sentences found. Exiting.")
        return

    print("\nSample sentence rows:")
    print(sentence_df[["conversation_id", "assistant_turn_index", "sentence_index", "sentence"]].head(5))

    print("\nLoading NeuroBiber...")
    tokenizer, model, device = load_neurobiber()
    print(f"Using device: {device}")

    print("Running Phase 2 clause candidate screening...")
    screened_df = run_clause_candidate_screening(
        sentence_df=sentence_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        prediction_batch_size=PREDICTION_BATCH_SIZE,
        keep_zero_feature_rows=False,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    screened_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Saved screened candidates to: {OUTPUT_FILE}")
    print(f"Candidate rows: {len(screened_df)}")

    if not screened_df.empty:
        print("\nComplement feature counts:")
        for feat in sorted(COMPLEMENT_FEATURES):
            print(f"{feat}: {int(screened_df[feat].sum())}")

        print("\nVerb-class feature counts:")
        for feat in sorted(VERB_CLASS_FEATURES):
            print(f"{feat}: {int(screened_df[feat].sum())}")

        print("\nSample candidate rows:")
        cols_to_show = [
            "conversation_id",
            "model",
            "assistant_turn_index",
            "sentence_index",
            "sentence",
            "is_candidate",
            "BIN_THVC",
            "BIN_THAC",
            "BIN_TOBJ",
            "BIN_TSUB",
            "BIN_PRIV",
            "BIN_PUBV",
        ]
        print(screened_df[cols_to_show].head(10))


if __name__ == "__main__":
    main()
