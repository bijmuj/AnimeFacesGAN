import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import imageio 

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)/127.5 - 1.
    return tf.image.resize(image, (64, 64))



def generate_and_save_images(model, epoch, test_input, out_path):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
      plt.axis('off')

  plt.savefig(out_path + 'image_at_epoch_{:03d}.png'.format(epoch))
#   plt.show()


def get_data(imgs_path, batch_size):
    dataset_path = glob.glob(imgs_path+'/*.jpg')
    dataset = tf.data.Dataset.from_tensor_slices(dataset_path)
    dataset = dataset.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    train_data = dataset.shuffle(2500).batch(batch_size)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_data


def make_gif(out_path):
    out_file = out_path + "faces.gif"
    with imageio.get_writer(out_file, mode='I') as writer:
        filenames = glob.glob(out_path + '*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)