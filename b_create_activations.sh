# for loop run create_activation_dataset2.py


model_name=version_19_setting_0
model_file=model_single_bucket
ckpt_name="epoch=499-step=3052000.ckpt"

# define variable length
#python create_fen_dataset.py
declare -i length=10000
for i in {0..100}
do  
    echo "python p_activation_dataset.py --start $((i*length)) --length $length --model_name $model_name --model_file $model_file --ckpt_name $ckpt_name"
    python p_activation_dataset.py --start $((i*length)) --length $length --model_name $model_name --model_file $model_file --ckpt_name $ckpt_name
done
