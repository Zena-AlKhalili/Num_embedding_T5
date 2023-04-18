# Num_embedding_T5
To Train:
```console
!python main.py  -m google/t5-small-lm-adapt -ht reg -em True -voc '{vocab_file_path}' -on '{output_model_name}' -e 200  -tf '{train_file_path}'  -df '{dev_file_path}'   --train
```
To predict:
```console
!python main.py' -m google/t5-small-lm-adapt -ht reg -em True -voc '{vocab_file_path}'  -omp '{output_model_path}'  -tsf '{test_file_path}'  --predict
```
