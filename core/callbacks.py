import tensorflow as tf
from settings import ws
import math


def get_log_dir(ts, exp, obj):
    return ws + "/logs/tensorflow_hub/" + exp + "/" + ts + "/" + obj


# Create tensorboard callback (functionized because need to create a new one for each model)
def create_tensorboard_callback(experiment_name, ts):
    log_dir = get_log_dir(ts, experiment_name, "tensorboard")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_images=True,
        histogram_freq=4
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_checkpoint_callback(ts, experiment_name):
    log_dir = get_log_dir(ts, experiment_name, "checkpoint")
    callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir,
        save_best_only=True,
        monitor="val_loss"
    )
    print(f"Saving callback log files to: {log_dir}")
    return callback


early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, ts, experiment_name, train, validation=None):
        super(TensorBoardImage, self).__init__()
        self.logdir = get_log_dir(ts, experiment_name, "images")
        self.train = train
        self.validation = validation
        print(f"Saving images  to: {self.logdir}")
        self.file_writer = tf.summary.create_file_writer(self.logdir)

    def on_batch_end(self, batch, logs):
        images_or_labels = 0  # 0=images, 1=labels
        img = self.train[batch][images_or_labels]
        # print("...OnBatchEnd: {}; got log keys: {} and img: {}".format(batch, logs,img.shape))
        # calculate epoch
        n_batches_per_epoch = self.train.samples / self.train.batch_size
        epoch = math.floor(self.train.total_batches_seen / n_batches_per_epoch)

        # since the training data is shuffled each epoch, we need to use the index_array to find something which uniquely
        # identifies the image and is constant throughout training
        first_index_in_batch = batch * self.train.batch_size
        last_index_in_batch = first_index_in_batch + self.train.batch_size
        last_index_in_batch = min(last_index_in_batch, len(self.train.index_array))
        img_indices = self.train.index_array[first_index_in_batch: last_index_in_batch]

        # convert float to uint8, shift range to 0-255
        img -= tf.reduce_min(img)
        img *= 255 / tf.reduce_max(img)
        img = tf.cast(img, tf.uint8)
        # print("...OnBatchEnd: {}; got log keys: {} and img: {}".format(batch, logs,img))

        with self.file_writer.as_default():
            # print("...writing files ....",img.shape)

            for ix, img in enumerate(img):
                img_tensor = tf.expand_dims(img, 0)  # tf.summary needs a 4D tensor
                # only post 1 out of every 1000 images to tensorboard
                # if (img_indices[ix] % 1000) == 0:
                # if (img_indices[ix] % batch_size/2) == 0:
                if (img_indices[ix] % 3) == 0:
                    # instead of img_filename, I could just use str(img_indices[ix]) as a unique identifier
                    # but this way makes it easier to find the unaugmented image
                    img_filename = self.train.filenames[img_indices[ix]]
                    # print("Image filename {} at index={}".format(img_filename,ix))
                    # print("Image",img_tensor)
                    tf.summary.image(img_filename, img_tensor, step=epoch)
