out_dir=("u01" "u02" "u03" "u04")

CUDA_VISIBLE_DEVICES=0 python run_multi_cho.py  \
  --do_train \
  --do_eval \
  --task_name ReClor \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --num_train_epochs 6 \
  --fp16 \
  --load_model_path microsoft/deberta-v3-base \
  --output_dir mrc/de-b/reclor/n

n=3
for ((i=0; i<=n; i++)); do
CUDA_VISIBLE_DEVICES=0 python run_multi_cho.py  \
  --do_train \
  --do_eval \
  --task_name ReClor \
  --learning_rate 1e-5 \
  --train_batch_size 16 \
  --num_train_epochs 6 \
  --fp16 \
  --load_model_path microsoft/deberta-v3-base \
  --output_dir mrc/de-b/reclor/${out_dir[$i]} \
  --load_state_dict clr/de-b/${out_dir[$i]}/2_pytorch_model.bin
done
