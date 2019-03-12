import os
import numpy as np
import hyperparams as hp

from tensor2tensor.data_generators import generator_utils


def _generator(input_path):
    for line in open(os.path.join(input_path, "metadata.csv"), "r"):
        uid, _, sent = line.split("|")
        mels = np.load(os.path.join(input_path, mels, "%s.npy" % uid))
        mags = np.load(os.path.join(input_path, mags, "%s.npy" % uid))
        yield {"uid": uid,
               "sent": sent,
               "mels": mels,
               "mags": mags}

def generate_tf_records(out_path="/home/abhishek/tmp/tmp/tf-records",input_path="/home/abhishek/tmp/LJ-Speech-1.1", num_shards=100):
    train_paths = generator_utils.train_data_filenames("lj_speech", out_path, num_shards)
    generator_utils.generate_files(_generator(input_path), train_paths)
