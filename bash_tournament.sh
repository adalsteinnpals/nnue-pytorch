
# declare ID variable
ID=5

../PycharmProjects/c-chess-cli/c-chess-cli -rounds 1 -concurrency 32 -each option.Hash=32 option.Threads=1 timeout=20 tc=100.0+0.04 -games 200 \
-openings file=./noob_3moves.epd order=random srand=64188023 \
-repeat \
-resign count=10 score=7000 \
-draw number=100 count=8 score=1 \
-pgn models/pgn/out_test_$ID.pgn 3 \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=always_queens option.EvalFile=/home/ap/nnue-pytorch/models/nn-epoch499_always_queens.nnue \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=regular_model option.EvalFile=/home/ap/nnue-pytorch/models/nn-epoch499_regular_model.nnue \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=no_bishop_pair_advantage option.EvalFile=/home/ap/nnue-pytorch/models/nn-epoch499_no_bishop_pair_advantage.nnue \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=no_queen_advantage option.EvalFile=/home/ap/nnue-pytorch/models/nn-epoch499_no_queen_advantage.nnue \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=no_knight_advantage option.EvalFile=/home/ap/nnue-pytorch/models/nn-epoch499_no_knight_pair_advantage.nnue \
-engine cmd=../PycharmProjects/Stockfish/src/stockfish name=master



../PycharmProjects/Ordo/ordo -p models/pgn/out_test_$ID.pgn -o models/ratings/ratings_$ID.txt