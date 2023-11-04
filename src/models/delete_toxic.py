from typing import List 
from utils import predict_toxicity
from tqdm import trange

def delete_toxic(df, 
                 threshold=0.5, 
                 size=20, 
                 output_path='/home/dmitrii/vscode_projects/PMLDL/Assignment1/data/output_files/delete_output.txt'
                 ):
    print(df.head())
    toxic_sentences = df.reference[:size].tolist()
    detoxified_text = []
    for i in trange(len(toxic_sentences)):
        words = toxic_sentences[i].split()
        toxic_scores = predict_toxicity(words)
        detoxified_sentence = " ".join([sentence for sentence, score in zip(words, toxic_scores) if score < threshold])
        detoxified_text.append(detoxified_sentence)
    with open(output_path, 'w') as file:
         file.write('\n'.join(detoxified_text))
    return detoxified_text
