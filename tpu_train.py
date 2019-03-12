import tensorflow as tf

import train

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("task_num", None,
                     "The type of task to run. 1 = Text2Mel, 2 = SSRN.")
flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

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


def model_fn_builder(task_num, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """
    	The `model_fn` for TPUEstimator.
    
    	From TF documentation:
    		https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args
    
       """
      
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
      
        #TODO(anon): figure out a way to pass/consume input here!
	# input_ids = features["input_ids"]
        # input_mask = features["input_mask"]
        # segment_ids = features["segment_ids"]
        # label_ids = features["label_ids"]
        # is_real_example = None
        # if "is_real_example" in features:
        #   is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        # else:
        #   is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
      
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
     
        graph = train.Graph(task_num)
        total_loss = graph.loss
        
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
    	output_spec = None
    	if mode == tf.estimator.ModeKeys.TRAIN:

    	  train_op = optimization.create_optimizer(
    	      total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    	  output_spec = tf.contrib.tpu.TPUEstimatorSpec(
    	      mode=mode,
    	      loss=total_loss,
    	      train_op=train_op,
    	      scaffold_fn=scaffold_fn)
	else:
	  raise Exception("Unsupported mode on tpu_train: %s" % mode)
	return output_spec


def input_fn_builder(input_path, num_epochs=500):
    def input_fn(params):
        batch_size = params["batch_size"]
        files = tf.data.Dataset.list_files(input_path)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=32)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parser_fn, num_parallel_calls=64)
        dataset = dataset.batch(batch_size)
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
  	    learning_rate=FLAGS.learning_rate,
  	    num_train_steps=num_train_steps,
  	    num_warmup_steps=num_warmup_steps,
  	    use_tpu=FLAGS.use_tpu)

    #TODO(anon): code up the estimator and input_fn
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    # if FLAGS.do_train:
    #   train_features = convert_examples_to_features(
    #       train_examples, label_list, FLAGS.max_seq_length, tokenizer)
    #   tf.logging.info("***** Running training *****")
    #   tf.logging.info("  Num examples = %d", len(train_examples))
    #   tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #   tf.logging.info("  Num steps = %d", num_train_steps)
    #   train_input_fn = input_fn_builder(
    #       features=train_features,
    #       seq_length=FLAGS.max_seq_length,
    #       is_training=True,
    #       drop_remainder=True)
    #   estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == "__main__":
    flags.mark_flag_as_required("task_num")
    tf.app.run()



