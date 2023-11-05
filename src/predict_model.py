from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from src.data.make_dataset import preprocess_dataset, preprocess_function_extra, get_dataset
from utils import compute_metric_with_extra, set_seeds, save_temp
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("model_checkpoint")
parser.add_argument("--n_rows", type=int, default=1500)

args = parser.parse_args()
model_checkpoint = args.model_checkpoint
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

predictions, labels, metrics = trainer.predict(tokenized_datasets["test"], metric_key_prefix="predict")
my_predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
print(metrics)
print(my_predictions)
save_temp("../data/output_files/t5small_output.txt", my_predictions)
