import os
import numpy as np
import hyperparams as hp
import tqdm
import data_load

from tensor2tensor.data_generators import generator_utils


def parse(example_proto):
    spec = {"sent": tf.VarLenFeature(tf.string),
            "mels": tf.VarLenFeature(tf.float32), 
            "mels_shape": tf.FixedLenFeature([2], tf.int64), 
            "mags": tf.VarLenFeature(tf.float32),
            "mags_shape": tf.FixedLenFeature([2], tf.int64) 
           }
    features = tf.parse_single_example(example_proto)
    mels = features["mels"]
    mels = tf.sparse_to_dense(mels.indices, mels.dense_shape, mels.values, default_value=0.0)
    features["mels"] = tf.reshape(mels, spec["mels_shape"])
    mags = features["mags"]
    mags = tf.sparse_to_dense(mags.indices, mags.dense_shape, mags.values, default_value=0.0)
    features["mags"] = tf.reshape(mags, spec["mags_shape"])
    return features

def _generator(input_path, char2idx):
    with open(os.path.join(input_path, "metadata.csv"), "r") as fp:
        lines = fp.readlines()

    for line in tqdm.tqdm(lines): 
        uid, _, sent = line.split("|")
        mels = np.load(os.path.join(input_path, "mels", "%s.npy" % uid))
        mags = np.load(os.path.join(input_path, "mags", "%s.npy" % uid))
        yield {"uid": uid,
               "sent": bytes(data_load.get_cleaned_char_id(sent, char2idx)),
               "mels": mels.tobytes(),
               "mels_shape": mels.shape,
               "mags_shape": mags.shape,
               "mags": mags.tobytes()}

def generate_tf_records(out_path="/home/abhishek/tmp/tf-records",input_path="/home/abhishek/tmp/LJSpeech-1.1", num_shards=100):
    train_paths = generator_utils.train_data_filenames("lj_speech", out_path, num_shards)
    char2idx, _ = data_load.load_vocab()
    generator_utils.generate_files(_generator(input_path, char2idx), train_paths)

if __name__ == "__main__":
    generate_tf_records()