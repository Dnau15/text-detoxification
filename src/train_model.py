import argparse
import torch
from data.make_dataset import preprocess_dataset, preprocess_function_extra, get_dataset
from models.delete_toxic import delete_toxic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,DataCollatorForSeq2Seq
from utils import compute_metric_with_extra, set_seeds

# set all seeds
set_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = [
    "t5-base",
    "s-nlp/t5-paranmt-detox",
    "s-nlp/bart-base-detox",
    "t5-small"
]

# Parse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("model", choices=model_names)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--toxic_threshold", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_input_length", type=int, default=128)
parser.add_argument("--max_target_length", type=int, default=128)
parser.add_argument("--test_size", type=float, default=0.15)
parser.add_argument("--n_rows", type=int, default=1500)

args = parser.parse_args()

model_checkpoint = args.model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data = preprocess_dataset()
ds_splits = get_dataset(n_rows=args.n_rows)
tokenized_datasets = ds_splits.map(preprocess_function_extra(tokenizer,model_checkpoint), batched=True)

source_lang = "toxic"
target_lang = "detoxic"

model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=3,
    num_train_epochs=args.epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metric_with_extra(tokenizer)
)
trainer.train()