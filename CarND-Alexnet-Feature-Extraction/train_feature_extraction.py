import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle


# TODO: Load traffic signs data.

training_file = "train.p"
nb_classes = 43

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X, y = train['features'], train['labels']


# TODO: Split data into training and validation sets.

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=42)

# TODO: Define placeholders and resize operation.

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))
one_hot = tf.one_hot(labels, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
def fully_connected(x, shape, mu, sigma, actiavation=True):
    W = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(shape[-1]))
    fc = tf.matmul(x, W) + b
    if actiavation:
        return tf.nn.relu(fc)
    else:
        return fc

# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8 = fully_connected(fc7, shape, 0, 0.1, False)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(fc8, one_hot)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(fc8, 1), tf.argmax(one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.


def evaluate(X_data, y_data, BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x_tf: batch_x, y_tf: batch_y, dropout_keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


EPOCHS = 1
BATCH_SIZE = 128
SAVE_FILE = 'alex_net'
train = True 

if train:
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})

            validation_accuracy = evaluate(X_validation, y_validation, BATCH_SIZE)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, SAVE_FILE)
        print("Model saved")