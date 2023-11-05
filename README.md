## Student
Dmitrii Naumov
d.naumov@innopolis.univercity
BS21DS01
## Basic commands:
* git clone 
* source ./venv/bin/activate
* pip install -r text-detoxification/requirements.txt
## To train model
python train_model.py model_checkpoint --batch_size --epochs --toxic_threshold --lr --n_rows --test_size --max_target_length --max_input_length --weight_decay
## To predict
python predict_model.py model_checkpoint --n_rows
## To use baseline
python delete_toxic_words.py --threshold --size
## To visualize
python visualization/visualization.py --output_path
