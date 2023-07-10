# declare ID variable
depth=$1


ID=$2


stockfish_path=/home/ap/Stockfish
nnue_path=/home/ap/nnue-pytorch
ordo_path=../PycharmProjects/Ordo/ordo
cli_path=../PycharmProjects/c-chess-cli/c-chess-cli
games=500


if [ ! -d "tournaments" ]; then
  mkdir tournaments
fi

if [ ! -d "tournaments/pgn" ]; then
  mkdir tournaments/pgn
fi

if [ ! -d "tournaments/ratings" ]; then
  mkdir tournaments/ratings
fi



# make all engines in folder play against each other
for file in production_models/*/serialized_models/nn-epoch399-step.nnue
do
    engine_name=$(basename $file)
    echo $file
    # get second folder from path
    model=$(echo $file | cut -d'/' -f2)

    # make string -engine cmd=$stockfish_path/src/stockfish name=$engine_name option.EvalFile=$nnue_path/serialized_models/$model/$engine_name depth=$depth
    run_command="-engine cmd=$stockfish_path/src/stockfish name=$model option.EvalFile=$nnue_path/$file depth=$depth"
    # add run command to command string
    echo $run_command
    command="$command $run_command"
done

echo $command
    

$cli_path -rounds 1 -concurrency 30 -each option.Hash=32 option.Threads=1 timeout=20 tc=100.0+0.04 -games $games \
-openings file=./noob_3moves.epd order=random srand=64188023 \
-repeat \
-resign count=10 score=7000 \
-draw number=100 count=8 score=1 \
-pgn tournaments/pgn/out_test_$ID.pgn 3 \
$command


$ordo_path -p tournaments/pgn/out_test_$ID.pgn -o tournaments/ratings/ratings_$ID.txt