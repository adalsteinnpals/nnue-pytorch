
import torch
import pandas as pd
import shelve
import features
import nnue_dataset
from tqdm import tqdm
import logging 
import numpy as np
from scipy.sparse import lil_matrix, vstack
import click
import os
import pickle as pkl

logging.basicConfig(format='%(asctime)s — %(name)s — %(levelname)s — %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)



def get_data(feature_set, fens):
    b = nnue_dataset.make_sparse_batch_from_fens(
        feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    return b

def get_single_activation(model, forward_mode, feature_set, fens):

    model.forward_mode = forward_mode

    b = get_data(feature_set, fens)
    
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = b.contents.get_tensors("cuda")
    out = model.forward(
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
        layer_stack_indices,
    )
    out_numpy = out.cpu().detach().numpy()

    del out    


    return out_numpy





# indicate start index and length
@click.command()
@click.option('--start', default=0, help='Start index')
@click.option('--length', default=100, help='Length of dataset')
@click.option('--model_string', default="version_19_setting_0", help='Name of log for model')
@click.option('--model_file', default="model", help='Name of log for model')
@click.option('--ckpt_name', default="epoch=499-step=3052000.ckpt", help='Name of log for model')
def main(start, length, model_string, model_file, ckpt_name):

    if model_file == "model_single_bucket":
        import model_single_bucket as M
    elif model_file == "model":
        import model_get_activations as M
    else:
        raise Exception("model_file not recognized")
    
    dir_folder = f"production_models/{model_string}/activations"
    dir_name = f"{dir_folder}/{ckpt_name.split('-')[0].replace('=', '_')}"

    # make dir if not exists
    if not os.path.exists(dir_folder):
        os.mkdir(dir_folder)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # load df
    df = pd.read_pickle("production_models/fen_df.pkl")

    # if start is larger than length of df, return
    if start > df.shape[0]:

        print("skipping because start is larger than length of df")
        return

    features_name = "HalfKAv2_hm^"
    feature_set = features.get_feature_set_from_name(features_name)

    nnue = M.NNUE(feature_set)
    nnue = nnue.load_from_checkpoint(f'production_models/{model_string}/checkpoints/{ckpt_name}', feature_set=feature_set)
    nnue = nnue.cuda()

    #nnue = torch.load(f'production_models/{model_string}/checkpoints/{ckpt_name}')
    #nnue.set_feature_set(feature_set)
    #nnue = nnue.cuda()

    #nnue = torch.load('logs/default/version_11/checkpoints/epoch=499-step=3051999.ckpt')

    #print(nnue.keys())
    #nnue.set_feature_set(feature_set)
    
    print('running')


    forward_modes = nnue.possible_forward_modes + [100]

    input_size = 46592

    for forward_mode in tqdm(forward_modes, leave=False):
        
        key_name = f"layer{forward_mode}"

        if forward_mode == 10:
            arrays = lil_matrix((length, input_size), dtype=bool)
        else:
            arrays = []



        for i, idx in enumerate(tqdm(range(start, min(start + length, df.shape[0])), leave=False, desc=f"forward mode: {forward_mode}")):

            fens = [df.loc[idx, "fen"]]

            out = get_single_activation(nnue, forward_mode, feature_set, fens)

            #print(forward_mode)
            #print(out)
            #print(out.shape)

            if forward_mode == 10:
                # convert out to scripy sparse matrix
                arrays[i] = out[0]




                del out 

            

            else:
                arrays.append(out[0])

        if forward_mode != 10:
            arrays = np.array(arrays)





        if start == 0:
            # pickle arrays
            with open(f"{dir_name}/{key_name}.pkl", "wb") as f:
                pkl.dump(arrays, f)
        else:

            # load arrays
            with open(f"{dir_name}/{key_name}.pkl", "rb") as f:
                arrays_old = pkl.load(f)


            # check if key is list, convert to list and append if not
            if forward_mode == 10:
                arrays_new = vstack([arrays_old, arrays])
            else:
                arrays_new= np.concatenate([arrays_old, arrays], axis=0)

            # pickle arrays
            with open(f"{dir_name}/{key_name}.pkl", "wb") as f:
                pkl.dump(arrays_new, f)



if __name__ == "__main__":
    main()
