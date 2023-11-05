from evaluate.visualization import radar_plot
import matplotlib.pyplot as plt
import pandas as pd
import argparse


""""
The following code uses metrics that was calculated on test that using different models.
"""
baseline_metrics = {"bleu":84.3,
                   "rouge1":0.951,
                   "rouge2":0.895,
                   "TER":7.66}

bert_metrics = {"bleu":67.6,
                   "rouge1":0.958,
                   "rouge2":0.907,
                   "TER":12.67}

bert_tuned_metrics = {"bleu":74,
                   "rouge1":0.956,
                   "rouge2":0.909,
                   "TER":11.01}

t5_metrics = {"bleu":18.4,
                   "rouge1":0.547,
                   "rouge2":0.311,
                   "TER":67.93}

t5_tuned_metrics = {"bleu":24.66,
                   "rouge1":0.579,
                   "rouge2":0.356,
                   "TER":63.113}

bart_metrics = {"bleu":23.7,
                   "rouge1":0.59,
                   "rouge2":0.369,
                   "TER":62.61}

indexes = [["Baseline", "BERT"], ["BERT_tuned", "T5"], ["T5_tuned", "Bart"]]
columns = [[baseline_metrics, bert_metrics], [bert_tuned_metrics, t5_metrics], [t5_tuned_metrics, bart_metrics]]

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_path", type=str, default="../../data/output_files")
args = parser.parse_args()


data = pd.read_table('../../data/raw/filtered.tsv')
figure, axis = plt.subplots(1,2, figsize=(10,6)) 
axis[0].hist(data.ref_tox)
axis[0].set_title("reference toxicity")
axis[1].hist(data.trn_tox)
axis[1].set_title("translated_toxicity")
plt.savefig(f"{args.output_path}/distribution.png")

i = 0
for column, index in zip(columns, indexes): 
    i += 1
    plot = radar_plot(data=column, model_names=index)
    plot.show()
    plot.savefig(f"{args.output_path}/metrics{i}.png")