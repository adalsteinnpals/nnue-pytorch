
# First
# conda create -n nnuepytorch python=3.10 -y
# conda activate nnuepytorch


# git clone -b development https://github.com/adalsteinnpals/nnue-pytorch.git


# remove build folder

rm -rf build

# remove libtraining_data_loader.so

rm -rf libtraining_data_loader.so

# install requirements

conda env create --file=environment.yml

#conda init bash

#conda activate nnuepytorch


# build training_data_loader

sh compile_data_loader.bat


wget https://github.com/official-stockfish/books/blob/master/noob_3moves.epd.zip?raw=true -O noob_3moves.epd.zip


# unzip noob_3moves.epd.zip

unzip noob_3moves.epd.zip
