import torch
import gc
import random
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer
import tqdm
from transformers import logging
import evaluate
from datasets import load_dataset, load_metric

logging.set_verbosity_error()



def set_seeds(seed):
    """
    This function set all seed.

    Args:
        seed (_type_): seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def predict_toxicity(texts, device='cpu', clf_name = 's-nlp/roberta_toxicity_classifier_v1'):
    """
    This function predict toxicity of the words

    Args:
        texts (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        clf_name (str, optional): _description_. Defaults to 's-nlp/roberta_toxicity_classifier_v1'.

    Returns:
        _type_: _description_
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = RobertaForSequenceClassification.from_pretrained(clf_name).to(device)
    clf_tokenizer = RobertaTokenizer.from_pretrained(clf_name)
    with torch.inference_mode():
        inputs = clf_tokenizer(texts, return_tensors='pt', padding=True).to(clf.device)
        out = torch.softmax(clf(**inputs).logits, -1)[:, 1].cpu().numpy()
    return out


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metric_with_extra(tokenizer):
    sacrebleu_metric = load_metric("sacrebleu")
    rouge_metric = evaluate.load('rouge')
    ter_metric = evaluate.load("ter")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        sacrebleu_result = sacrebleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        ter_result = ter_metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {"bleu": sacrebleu_result["score"],
                "rouge1": rouge_result["rouge1"],
                 "rouge2": rouge_result["rouge2"],
                 "TER": ter_result["score"]}
        print(result)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    return compute_metrics

def save_temp(path, text):
    with open(path, "w", encoding="UTF-8") as file:
        file.write("\n".join(text))