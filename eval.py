import tensorflow as tf
import numpy as np
import os
import re
import data_helpers


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("test_dir", "data/test.csv", "Path of test data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("visualize", True, "Save the html visualization code")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


def eval():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.test_dir)

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_eval = np.array(list(vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]

            A = graph.get_operation_by_name("self-attention/attention").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(zip(x_eval, x_text)), FLAGS.batch_size, 1, shuffle=False)

            if FLAGS.visualize:
                f = open('visualize.html', 'w')
                f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')

            # Collect the predictions here
            all_predictions = []
            tokenizer = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
            for batch in batches:
                x_batch, text_batch = zip(*batch)

                batch_predictions, attention = sess.run([predictions, A], {input_text: x_batch})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

                if FLAGS.visualize:
                    f.write('<div style="margin:25px;">\n')
                    for k in range(len(attention[0])):
                        f.write('<p style="margin:10px;">\n')
                        ww = tokenizer.findall(text_batch[0])
                        print(attention[0][k])
                        for j in range(len(attention[0][0])):
                            alpha = "{:.2f}".format(attention[0][k][j])
                            if len(ww) > j:
                                w = ww[j]
                            else:
                                break
                            f.write(f'\t<span style="margin-left:3px;background-color:rgba(255,0,0,{alpha})">{w}</span>\n')
                        f.write('</p>\n')
                    f.write('</div>\n')

            if FLAGS.visualize:
                f.write('</body></html>')
                f.close()

            correct_predictions = float(sum(all_predictions == y_eval))
            print("\nTotal number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()