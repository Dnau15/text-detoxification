First of all, I want to remind that I use small and preprocessed version of the initial dataset because of capacity constraints.
#### Baseline: Dictionary based
One of the easiest and  understandable solutions is to identify toxic words in each sentence and then just delete them. Let's create toxic words identifier. I used  RobertaForSequenceClassification, RobertaTokenizer from transformers for evaluating toxicity of each word in a sentence. All in all, model works well, the main goal was achieved, but meaning of many sentences was lost.
#### Default BERT 
The second idea is using pretrained BERT for masking language modeling. However, I do not BERT to work as usual. BERT usually replace 15% of words with "[MASK]" token. Instead, I want to replace toxic words with "[MASK]" token. I use the following algorithm:
* evaluate toxicity of each word (using RobertaForSequenceClassification)
* replace each toxic word with "[MASK]" token
* use BERT to predict "[MASK]" token
* replace toxic word with predicted word
#### Fine-tuned BERT
The third idea is fine-tuning BERT from the second idea. I want to fine-tune BERT on translated part of the dataset. My idea is the following: usually reference text and translated text are similar. I believe that fine-tuning BERT on translated dataset will increase performance of the model because BERT will learn dependencies between words and will predict the most appropriate word instead of "[MASK]" token. For fine-tuning I will replace 15% of translated text with "[MASK]" token. 
#### T5
The 4-th idea is using T5 model. Firstly, I want to train T5-base model and then find model with similar task and try to fine-tune it. The idea is pretty obvious. Just translate reference sentence (in my case it would be toxic-sentence) to the non-toxic sentence.
#### T5 fine-tuned
As I said, 5-th idea is using T5 model that was already trained on the similar task. Model s-nlp/t5-paranmt-detox is suitable in my case. I want to fine-tune this model for 2 epochs
#### Bart
The last idea is using Bart model. The idea is the same as with T5 fine-tuned model. I choose s-nlp/bart-base-detox.
