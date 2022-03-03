# Emotion Recognition
This repo contains the code for the paper: 

***Speech Emotion Recognition with Multi-task Learning, X. Cai et al., INTERSPEECH 2021***

The code is based on https://github.com/huggingface/transformers/tree/master/examples/research_projects/wav2vec2.

## Files and folders
* paper_slides/: the paper and corresponding slides.
* model.py: the Wav2vec-2.0 model that inherites from Huggingface's Wav2vec-2.0 model, with a classification head in addition to the CTC head.
* run_emotion.py: the main python code that could runs the emotion recognition task.
* run.sh: the script to test running.
* iemocap/: the processed iemocap data pointers, split into 10 folds, while each fold has train.csv and test.csv. The original wavs are not here, please obtain from https://sail.usc.edu/iemocap/.
* requirements.txt: required packages to be installed.

## Set up environment
```bash
pip install -r requirements.txt
```

You might also need to install libsndfile:
```bash
sudo apt-get install libsndfile1-dev
```
Or refer to https://github.com/libsndfile/libsndfile.

## Prepare datasets
1. Obtain IEMOCAP dataset from https://sail.usc.edu/iemocap/.
2. Extract and save wav files at some path, assuming named as /wav_path/.
3. Replace the '/path_to_wavs' text in ./iemocap/\*.csv, with the actual path just saved all the wav files. You can use the following command.
```bash
for f in iemocap/*.csv; do sed -i 's/\/path_to_wavs/\/wav_path/' $f; done
```

Note: The iemocap/*.csv has 20 files, corresponding to the data split into 10 folds, according to session ID (01F, 01M, ..., 05F, 05M). For each fold, use the other 9 sessions as training, and test on the selected session. For example, for the fold 01F, we use 01F as test set and remaining 9 sessions as training set. Two csv files for each fold, one for training and one for testing. The names are: iemocap_01F.train.csv and iemocap_01F.test.csv. The csv file has 3 columns: file, emotion, text. The column 'file' indicates where to store the wav file; the column 'emotion' is the emotion label (we use 4 labels: e0, e1, e2, e3); the column 'text' is for transcript. For example:
```bash
file,emotion,text
/path/to/Ses01F_impro01_F000.wav,e0,"EXCUSE ME ."
/path/to/Ses01F_impro01_F001.wav,e0,"YEAH ."
...
```

## Minimum effort to run
```bash
bash run.sh
```
This will run the code and generates results in output/tmp/ folder, while cache files are stored in cache/.
The model = wav2vec2-base, alpha = 0.1, LR = 5e-5, effective batch size = 8, total train epochs = 200.
The 01F split will be used as testing and remaining will be used as training.

NOTE: At around 40 epochs, the eval acc should already reaches a fairly good results. But due to the learning rate at this time is still high, there will be fluctuations. Nevertheless, verifying or early stopping could save time and get a reasonable good model.

WARNING: If running on 1 single GPU, 200 epochs will take days to finish. To speed up, consider using multiple GPUs. By default, the code use all GPUs in the system.

## For inference 
After training, you can run the inference code, using the saved model in output/tmp (or providing another path with a saved model):
```bash
bash prediction.sh output/tmp
```
This will generate a classification result, for the 01F_test split, in output/predictions/tmp. Details can be found in the script.
NOTE: If you want to use your own inference data (prepared in a csv file), please modify the load_dataset() part in run_emotion.py.

## Reproduce using checkpoints
To reproduce the paper's results using the cheeckpoints:
1. Download the checkpoints from https://drive.google.com/drive/folders/1Ndybde47HDy8O7aiNvT14pT7FqkqOsSX?usp=sharing, save them at ./ckpts/
2. Run the following:
```bash
for s in 01F 01M 02F 02M 03F 03M 04F 04M 05F 05M; do bash reproduce.sh ckpts/$s/ $s > output/$s.eval; done
```
This will generate evaluation results using the downloaded checkpoints. The evaluation results are saved in output/ folder, named as 01F.eval, 01M.eval, etc.
The last line in each *.eval file, indicates number of correct predictions and accuracy for that folder (10-folds in total).
If you sum the correct number and divide by 5531 (total utterance number), you should get the accuraccy slightly above 0.78.

## Important parameters
Key parameters:
* MODEL : wav2vec2-base / wav2vec2-large-960h.
* ALPHA : loss = ctc + alpha * cls, 0.1 would be good enough for wav2vec2-base, 0.01 for wav2vec2-large-960h.
* LR : learning rate, recommended 5e-5 when effective batch size == 8.
* ACC : accumulated batch size. The effective batch size = batch_per_gpu * gpu_num * acc.
* WORKER_NUM : the number of cpu for data preprocessing, please set to the maximum cpu number in the machine.
* --num_train_epochs : number of training epochs, recommended > 100.
* --split_id : the split partition used for testing, values are 01F 01M 02F 02M 03F 03M 04F 04M 05F 05M. The reamining partitions are used for training.

Parameters not recommended:
* --group_by_length : this will significantly slow down data preprocessing step, but potentially improve training efficiency.

