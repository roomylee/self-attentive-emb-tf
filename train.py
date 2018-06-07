import tensorflow as tf
import numpy as np
import os
import datetime
import time
from self_attention import SelfAttention
import data_helpers


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_dir", "data/train.csv", "Path of train data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 50, "Max sentence length in train/test data (Default: 50)")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer (Default: 256)")
tf.flags.DEFINE_integer("d_a_size", 350, "Size of W_s1 embedding (Default: 350)")
tf.flags.DEFINE_integer("r_size", 30, "Size of W_s2 embedding (Default: 30)")
tf.flags.DEFINE_integer("fc_size", 2000, "Size of fully connected layer (Default: 2000)")
tf.flags.DEFINE_float("p_coef", 1.0, "Coefficient for penalty (Default: 1.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (Default: 10)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


def train():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.train_dir)

    # Build vocabulary
    # Example: x_text[3] = "A misty ridge uprises from the surge."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = SelfAttention(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                d_a_size=FLAGS.d_a_size,
                r_size=FLAGS.r_size,
                fc_size=FLAGS.fc_size,
                p_coef=FLAGS.p_coef
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(model.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                # Train
                feed_dict = {
                    model.input_text: x_batch,
                    model.input_y: y_batch
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    batches_dev = data_helpers.batch_iter(
                        list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    # Evaluation loop. For each batch...
                    loss_dev = 0
                    accuracy_dev = 0
                    cnt = 0
                    for batch_dev in batches_dev:
                        x_batch_dev, y_batch_dev = zip(*batch_dev)

                        feed_dict_dev = {
                            model.input_text: x_batch_dev,
                            model.input_y: y_batch_dev
                        }

                        summaries_dev, loss, accuracy = sess.run(
                            [dev_summary_op, model.loss, model.accuracy], feed_dict_dev)
                        dev_summary_writer.add_summary(summaries_dev, step)

                        loss_dev += loss
                        accuracy_dev += accuracy
                        cnt += 1

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_dev / cnt, accuracy_dev / cnt))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
