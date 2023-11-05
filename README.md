## Student
Dmitrii Naumov
d.naumov@innopolis.univercity
BS21DS01
## Basic commands:
* git clone 
* source ./venv/bin/activate
* pip install -r text-detoxification/requirements.txt
* python train_model.py model_checkpoint --batch_size --epochs --toxic_threshold --lr --n_rows --test_size --max_target_length --max_input_length --weight_decay
* python predict_model.py model_checkpoint --n_rows
* python delete_toxic_words.py --threshold --size
* python visualization/visualization.py --output_path
