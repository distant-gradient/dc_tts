import os
import numpy as np
import hyperparams as hp
import tqdm
import data_load
import tensorflow as tf

from tensor2tensor.data_generators import generator_utils  # TODO(anon): Remove this dependency



def _generator(input_path, char2idx):
    with open(os.path.join(input_path, "metadata.csv"), "r") as fp:
        lines = fp.readlines()
    # lines = lines[:200] # debug cap

    for line in tqdm.tqdm(lines): 
        uid, _, sent = line.split("|")
        mels = np.load(os.path.join(input_path, "mels", "%s.npy" % uid))
        mags = np.load(os.path.join(input_path, "mags", "%s.npy" % uid))
        yield {"uid": uid,
               "sent": bytes(data_load.get_cleaned_char_ids(sent, char2idx)),
               "mels": mels.flatten().tolist(),
               "mels_shape": mels.shape,
               "mags_shape": mags.shape,
               "mags": mags.flatten().tolist()}

def generate_tf_records(out_path="/home/abhishek/tmp/tf-records",input_path="/home/abhishek/tmp/LJSpeech-1.1", num_shards=100):
    train_paths = generator_utils.train_data_filenames("lj_speech", out_path, num_shards)
    char2idx, _ = data_load.load_vocab()
    generator_utils.generate_files(_generator(input_path, char2idx), train_paths)

if __name__ == "__main__":
    generate_tf_records()
