if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi



seq_len=336
model_name=HDMixer

dir=./logs/LongForecasting/
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
patch_len=16
stride=8

pred_len=96
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.8\
    --e_layers 1 \
    --batch_size 256\
    --learning_rate 5e-4\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 20\
    --train_epochs 50\
    --gpu 0\
    --itr 1\
    >$dir/$data_name'_'$seq_len'_'$pred_len.log 

pred_len=192
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.8\
    --e_layers 1 \
    --batch_size 256\
    --learning_rate 5e-4\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 20\
    --train_epochs 50\
    --gpu 0\
    --itr 1\
    >$dir/$data_name'_'$seq_len'_'$pred_len.log 

pred_len=336
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.8\
    --e_layers 2 \
    --batch_size 512\
    --learning_rate 5e-3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 20\
    --train_epochs 50\
    --gpu 0\
    --itr 1\
    >$dir/$data_name'_'$seq_len'_'$pred_len.log 

pred_len=720
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 32 \
    --dropout 0.8\
    --e_layers 2 \
    --batch_size 512\
    --learning_rate 5e-3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 20\
    --train_epochs 50\
    --gpu 0\
    --itr 1\
    >$dir/$data_name'_'$seq_len'_'$pred_len.log 