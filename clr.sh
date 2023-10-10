train_on=("u01" "u02" "u03" "u04")
eval_on=("u01" "u02" "u03" "u04")
out_dir=("u01" "u02" "u03" "u04")

CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py  \
  --do_train \
  --do_eval \
  --data_dir boolkill/u \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --num_train_epochs 3 \
  --fp16 \
  --load_model_path microsoft/deberta-v3-base \
  --output_dir clr/de-b/${out_dir[0]} \
  --train_on ${train_on[0]} \
  --eval_on ${eval_on[0]}

n=3
for ((i=1; i<=n; i++)); do
CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py  \
  --do_train \
  --do_eval \
  --data_dir boolkill/u \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --num_train_epochs 3 \
  --fp16 \
  --load_model_path microsoft/deberta-v3-base \
  --output_dir clr/de-b/${out_dir[$i]} \
  --train_on ${train_on[$i]} \
  --eval_on ${eval_on[$i]} \
  --load_state_dict clr/de-b/${out_dir[$i-1]}/2_pytorch_model.bin
done

n=3
for ((i=0; i<=n; i++)); do
CUDA_VISIBLE_DEVICES=0 python run_sent_clas.py  \
  --do_eval \
  --data_dir boolkill/u \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --num_train_epochs 3 \
  --fp16 \
  --load_model_path microsoft/deberta-v3-base \
  --output_dir clr/de-b/${out_dir[$i]} \
  --eval_on u58 \
  --load_state_dict clr/de-b/${out_dir[$i]}/2_pytorch_model.bin
done
