from model.vgg import vgg_custom
import util.dataLoader4 as dataLoader
import tensorflow as tf
import argparse
import numpy as np


def get_next_batch(dataset, batch_size):
    c0_image, c0_label = dataset.class0.next_batch(
            batch_size=batch_size)
    c1_image, c1_label = dataset.class1.next_batch(
        batch_size=batch_size)
    
    train_img = np.concatenate((c0_image, c1_image), axis=0)
    train_lbl = np.concatenate((c0_label, c1_label), axis=0)
    
    perm = np.arange(batch_size+batch_size)
    np.random.shuffle(perm)
    train_img = train_img[perm]
    train_lbl = train_lbl[perm]    
    return train_img, train_lbl 

def main(args):
    logdir = "./logs/" + args.logdir + "/"

    # Parameters
    #learning_rate = 0.1
    learning_rate = args.learnRate
    decay_rate = 0.96
    batch_size = int(args.batchSize / 2)
    display_step = 10
    epochSize = 40
    #training_iters = 1000*epochSize*batch_size
    

    # Network Parameters
    imagesize = 224
    img_channel = 3
    n_classes = 2
    dropout = 0.8  # Dropout, probability to keep units
    dataset = dataLoader.read_data(imagesize, imagesize, img_channel)
    len_class0 = len(dataset.class0.labels)
    len_class1 = len(dataset.class0.labels)
    len_dataset = len_class0 + len_class1    
    

    training_iters = args.iteration * int( len_dataset / args.batchSize )
    
    # tf Graph input
    x = tf.placeholder("float", [None, imagesize, imagesize, img_channel])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder("float")  # dropout (keep probability)

    # Construct model
    vgg = vgg_custom()
    pred = vgg.model(x, keep_prob, n_classes, img_channel, numOfGpu = 1)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    global_step = tf.Variable(0, trainable=False)
    
    lr = tf.train.exponential_decay(
        learning_rate, global_step, training_iters,
        decay_rate=decay_rate, staircase=True)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=lr, momentum = 0.9).minimize(cross_entropy, global_step=global_step)

    saver = tf.train.Saver()
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("keep_prob", keep_prob)
    tf.add_to_collection("pred", pred)
    tf.add_to_collection("accuracy", accuracy)

    # Initializing the variables
    init = tf.global_variables_initializer()
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('global_step', global_step)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('lr', lr)

    tf.summary.histogram('y_hist', y)
    tf.summary.histogram('pred_hist', pred)

    tf.summary.image('x', x)

    tf.summary.histogram('cross_entropy_hist', cross_entropy)
    tf.summary.histogram('accuracy_hist', accuracy)

    merged = tf.summary.merge_all()
    # Launch the graph
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init)
        step = 1
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            
            train_img, train_lbl = get_next_batch(dataset, batch_size)
            
            feedDict = {x: train_img, y : train_lbl, keep_prob : dropout}
            summary, _ = sess.run([merged, optimizer], feed_dict=feedDict)

            if step % display_step == 0:
                # Calculate batch accuracy
                # train_img, train_lbl = get_next_batch(dataset, batch_size)                                
                
                acc, cross = sess.run([accuracy, cross_entropy], feed_dict={
                    x: train_img, y: train_lbl, keep_prob: 1.})                        
    
                writer.add_summary(summary, step)
                saver.save(sess, logdir + 'model.ckpt', global_step=step * batch_size)
                print("Iter {} Minibatch Loss= {}, Training Accuracy={}".format(step, cross, acc))

            step += 1
                    
        acc, cross = sess.run([accuracy, cross_entropy], feed_dict={
            x: dataset.valid.images, y: dataset.valid.labels, keep_prob: 1.})
        print("valid Minibatch Loss= " +
                      "{}".format(cross) + ", valid Accuracy= " + "{}".format(acc))

        # print "Testing Accuracy:", sess.run(accuracy, feed_dict={x:
        # dataset.test.images, y: dataset.test.labels, keep_prob: 1.})

        # print "predict test  class:", sess.run([pred], feed_dict={x:
        # dataset.test.images[0:3], keep_prob: 1.})[0]

        # print "predict valid class:", sess.run([pred], feed_dict={x:
        # dataset.valid.images, keep_prob: 1.})[0] * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="vgg",
                        help="path to log dir ")
    parser.add_argument("--batchSize", type=int,
                        default=80, help="size of bastch run")
    parser.add_argument("--iteration", type=int,
                        default=1000, help="count of iteration")
    parser.add_argument("--learnRate", type=float,
                        default=0.001, help="learn rate")
    args = parser.parse_args()
    main(args)
