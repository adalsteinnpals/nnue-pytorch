import nnue_dataset
import chess

def make_fen_batch_provider(data_path, batch_size, train_setting):
    return nnue_dataset.FenBatchProvider(data_path, True, 1, batch_size, False, 10, train_setting=train_setting)



if __name__ == '__main__':
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, 2)

    fen_batch = next(dataloader)
    for fen in fen_batch[:10]:
        board = chess.Board(fen)
        print('-------------------')
        if 'q' in fen: print('black has queen')
        if 'Q' in fen: print('white has queen')
        print('number of white bishops: ', fen.split(' ')[0].count('B'))
        print('number of black bishops: ', fen.split(' ')[0].count('b'))
        print(fen)
        print(board)