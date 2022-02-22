export CKPT=$1 # the path of trained model
export SPLIT=$2 # the iemocap split, e.g. 01F, 01M
mkdir -p output/predictions
export SAVE_PATH=output/reproduce/$(basename $CKPT)

python run_emotion.py \
--dataset_name emotion \
--split_id $SPLIT \
--output_dir=$SAVE_PATH \
--overwrite_output_dir \
--num_train_epochs=200 \
--warmup_ratio 0.1 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps=1 \
--alpha 0.1 \
--evaluation_strategy="steps" \
--save_total_limit="1" \
--do_eval \
--output_file $SAVE_PATH/predictions.txt \
--save_steps="500" \
--eval_steps="5" \
--logging_steps="5" \
--logging_dir="log" \
--learning_rate=0.1 \
--model_name_or_path $CKPT \
--tokenizer facebook/wav2vec2-base \
--preprocessing_num_workers=20 \
--fp16 \
--dataloader_num_workers 4

