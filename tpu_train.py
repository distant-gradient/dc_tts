import tensorflow as tf

import train
from hyperparams import Hyperparams as hp

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("task_num", None,
                     "The type of task to run. 1 = Text2Mel, 2 = SSRN.")
flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
        "input_file_pattern", None,
        "The input dir with TFRecords.")

flags.DEFINE_integer(
        "train_batch_size", 32, "Size of training batch")

flags.DEFINE_integer(
        "iterations_per_loop", 10, "This is the number of train steps running in TPU system before returning to CPU host for each Session.run")

flags.DEFINE_integer(
        "save_checkpoints_steps", 10, "Num steps after which to checkpoint")


flags.DEFINE_integer(
        "eval_batch_size", 32, "Size of eval batch")

flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained model).")


def model_fn_builder(task_num, init_checkpoint, use_tpu):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """
    	The `model_fn` for TPUEstimator.
    
    	From TF documentation:
    		https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args
    
       """
      
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
      
        sents, mels, mags = features["sent"], features["mels"], features["mags"] 
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
     
        graph = train.Graph(task_num, L=sents, mels=mels, mags=mags, use_tpu=use_tpu)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
          (assignment_map, initialized_variable_names
          ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
          if use_tpu:
      
            def tpu_scaffold():
              tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
              return tf.train.Scaffold()
      
            scaffold_fn = tpu_scaffold
          else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
    	    mode=mode,
    	    loss=graph.loss,
    	    train_op=graph.train_op,
    	    scaffold_fn=scaffold_fn)
	return output_spec
    return model_fn


def parse_fn(example_proto):
    spec = {"sent": tf.VarLenFeature(tf.int64),
            "mels": tf.VarLenFeature(tf.float32), 
            "mels_shape": tf.FixedLenFeature([2], tf.int64), 
            "mags": tf.VarLenFeature(tf.float32),
            "mags_shape": tf.FixedLenFeature([2], tf.int64) 
           }
    features = tf.parse_single_example(example_proto, features=spec)
    mels = features["mels"]
    mels = tf.sparse_to_dense(mels.indices, mels.dense_shape, mels.values, default_value=0.0)
    features["mels"] = tf.reshape(mels, [-1, hp.n_mels])
    mags = features["mags"]
    mags = tf.sparse_to_dense(mags.indices, mags.dense_shape, mags.values, default_value=0.0)
    features["mags"] = tf.reshape(mags, [-1, hp.n_fft//2+1])
    sent = features["sent"]
    features["sent"] = tf.sparse_to_dense(sent.indices, sent.dense_shape, sent.values, default_value=0)
    return features

def input_fn_builder(input_path, num_epochs=500):
    def input_fn(params):
        batch_size = params["batch_size"]
        files = tf.data.Dataset.list_files(input_path)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_fn, num_parallel_calls=64)
        dataset = dataset.padded_batch(batch_size, [-1, -1])
        dataset = dataset.prefetch(2)
        return dataset
    return input_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_num = FLAGS.task_num
    
    # get the TPU run_config
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # Assign the model_fn -- function that takes in (features, labels, mode, params)
    # see https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args
    model_fn = model_fn_builder(
  	    task_num=task_num,
  	    init_checkpoint=FLAGS.init_checkpoint,
  	    use_tpu=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        params={},
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    train_input_fn = input_fn_builder(
        input_path=FLAGS.input_file_pattern)
    estimator.train(input_fn=train_input_fn, max_steps=100000)

if __name__ == "__main__":
    flags.mark_flag_as_required("task_num")
    tf.app.run()



