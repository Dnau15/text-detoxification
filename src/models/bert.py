from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer
import torch
from utils import predict_toxicity
import numpy as np


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def mask_toxic(sentence, threshold=0.3):
    words = sentence.split()
    probabilities = predict_toxicity(words)
    text_prep = []
    toxic_indexes = []
    index = 0
    for _word, _prob in zip(words, probabilities):
        if _prob > threshold:
            toxic_indexes.append(index)
            text_prep.append("[MASK]")
        else:
            text_prep.append(_word)
        index += 1
    text_prep = " ".join(text_prep)
    tokenized = tokenizer(text_prep, return_tensors="pt")
    return tokenized, toxic_indexes


def get_top_k_pred(inputs, k=4):
    with torch.no_grad():
        logits = model(**inputs).logits
    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index]
    flat = predicted_token_id.flatten()
    top_k = np.argpartition(flat, -k)[-k:]
    top_k_sorted = top_k[np.argsort(flat[top_k])]
    unmasked = [tokenizer.decode(top) for top in top_k_sorted]
    #non_toxic = [x if predict_toxicity(x)[0] < 0.3 else " " for x in unmasked]
    return unmasked, top_k_sorted


def bert_detoxify(sentence, threshold=0.5, k=4):
    masked, toxic_indexes = mask_toxic(sentence)
    while len(toxic_indexes) > 0:
        idx = toxic_indexes.pop(0)
        unmasked, top_k_pred = get_top_k_pred(masked)
        print("TOP K:", top_k_pred)
        sentence[idx] = unmasked[-1]
        if len(toxic_indexes) > 0:
            masked = tokenizer(sentence, return_tensors="pt")
        #top_k_pred[0]
    #masked[mask_index] = top_k_pred[-1]
    return masked

