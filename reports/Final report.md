### Introduction
The main task of this assignment is text detoxification. Text detoxification is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.
### Data analysis
![[Pasted image 20231103133222.png]]
The dataset consists of 6 columns and 577779 rows.
Columns:
* reference - initial sentence
* translation - translated sentence
* similarity - cosine of the texts
* length_diff - relative length difference between texts
* ref_tox - toxicity score of initial sentence
* trn_tox  - toxicity score of translated sentence
The main observation of this part is that reference sentence can be less toxic than translation sentence. It means that we have at least 2 options:
* We can just remove all such cases (Where reference sentence is more toxic than translated)
* We can rearrange translated sentence and reference sentence
I found some rows with inappropriate data. I mean not really toxic data for reference and not really non-toxic data for translation. I want to delete such columns because it negatively affects on the training stage.
### Model Specification
### Training process
### Evaluation
### Results
