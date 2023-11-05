from typing import List 
from utils import predict_toxicity
from tqdm import trange
import evaluate
from datasets import load_metric


def delete_toxic(df, 
                 threshold=0.5, 
                 size=20, 
                 output_path='../data/output_files/delete_output.txt'
                 ):
    sacrebleu_metric = load_metric("sacrebleu")
    rouge_metric = evaluate.load('rouge')
    ter_metric = evaluate.load("ter")
    print(df.head())
    toxic_sentences = df.reference[:size].tolist()
    toxic_sentences_list = [[t] for t in toxic_sentences]
    detoxified_text = []
    result = {
        "bleu": 0,
        "rouge1": 0,
        "rouge2": 0,
        "TER": 0
    }
    n = len(toxic_sentences)
    for i in trange(len(toxic_sentences)):
        words = toxic_sentences[i].split()
        toxic_scores = predict_toxicity(words)
        detoxified_sentence = " ".join([sentence for sentence, score in zip(words, toxic_scores) if score < threshold])
        detoxified_text.append(detoxified_sentence)

    result["bleu"] = sacrebleu_metric.compute(predictions=detoxified_text, references=toxic_sentences_list)["score"]
    rouge_score = rouge_metric.compute(predictions=detoxified_text, references=toxic_sentences_list)
    result["rouge1"] = rouge_score["rouge1"]
    result["rouge2"] = rouge_score["rouge2"]
    result["TER"] = ter_metric.compute(predictions=detoxified_text, references=toxic_sentences_list)["score"]
    # Let's print last 5 predictions
    print(result)
    with open(output_path, 'w') as file:
         file.write('\n'.join(detoxified_text))
    return detoxified_text
