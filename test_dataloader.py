import nnue_dataset
import chess
import pytest

def make_fen_batch_provider(data_path, batch_size, train_setting):
    return nnue_dataset.FenBatchProvider(data_path, True, 1, batch_size, False, 10, train_setting=train_setting)


# Custom Training Setting
# 0 = regular
# 1 = always with queens
# 2 = never only one with bishop pair
# 3 = never only one with queen
# 4 = never only one with knight pair


def test_train_setting_1():
    """
    When train_setting = 1, the dataloader should only return positions where both have queens.
    """
    train_setting = 1
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        assert 'q' in fen and 'Q' in fen


def test_train_setting_2():
    """
    When train_setting = 2, the dataloader should never return positions where only one side has two bishops.
    """

    train_setting = 2
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not (pos_string.count('B') == 2 and pos_string.count('b') != 2) 
        assert not (pos_string.count('B') != 2 and pos_string.count('b') == 2)



def test_train_setting_3():
    """
    When train_setting = 3, the dataloader should never return positions where only one side has a queen.
    """
    
    train_setting = 3
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not ((pos_string.count('q') == 0) and (pos_string.count('Q') > 0))
        assert not ((pos_string.count('q') > 0) and (pos_string.count('Q') == 0))



def test_train_setting_4():
    """
    When train_setting = 4, the dataloader should never return positions where only one side has a knight pair.
    """

    train_setting = 4
    dataloader = make_fen_batch_provider('../PycharmProjects/data/training_data.binpack', 1000, train_setting)

    fen_batch = next(dataloader)
    for fen in fen_batch:
        pos_string = fen.split(' ')[0]
        assert not (pos_string.count('N') == 2 and pos_string.count('n') != 2) 
        assert not (pos_string.count('N') != 2 and pos_string.count('n') == 2)






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