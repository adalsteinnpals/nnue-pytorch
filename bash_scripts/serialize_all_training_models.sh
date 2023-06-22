# serialize all models in folder

folder=version_20

mkdir -p serialized_models/$folder
for file in logs/lightning_logs/$folder/checkpoints/epoch=*.ckpt
do
    epochnumber=$(basename $file .ckpt | cut -d'=' -f2)
    file_name="nn-epoch$epochnumber.nnue"
    python serialize.py $file serialized_models/$folder/$file_name
done