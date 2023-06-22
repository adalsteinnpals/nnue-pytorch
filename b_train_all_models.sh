

# Loop over all input arguments

for i in "$@"
do
bash train_model.sh $i
done

