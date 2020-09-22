# Anime Faces GAN

An implementation of DCGAN in tensorflow v2.3.0 trained on the  [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset).

The results from training for 50 epochs on a 1060 Max-Q are as follows:
![Gif](./data/out/faces.gif)

The default hyper-parameters were:

- Image size: 64x64
- Encoding size: 100
- Batch size: 100
- Learning rate: 2e-4
- Optimizer: Adam
- Beta_1: 0.5
- Epochs: 50

The images used for the gif were all generated from the same seed(1234).

To train the model use the following command:

```
python train.py [-i [path to checkpoint]] [-o [output path]] [-c [path to checkpoint]] [-b [batch size]] [-e [epochs]] [-l [learning rate]]
```

All the arguments are optional with default arguments being:

- -i ./data/images/
- -o ./data/out/
- -c ./ckpt/
- -b 100
- -e 50
- -l 0.0002

A directory for the checkpoints will need to be created if the default checkpoint path is used.

BinaryCrossentropy loss was replaced with the following losses for training:
```
def discriminator_loss(real_output, fake_output):
    real_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(real_output)))
    fake_loss = tf.math.negative(tf.math.reduce_mean(tf.math.log(1. - fake_output)))
    return real_loss, fake_loss
    

def generator_loss(fake_output):
    return tf.math.negative(tf.math.reduce_mean(tf.math.log(fake_output)))
```

Here, we used `max log(D)` to optimize the Generator instead of `min log(1 - D)`.

Flipping of the labels was not included as it caused the Discriminator loss to fall to near 0 and no training happened. Training for longer periods of time might benefit more from this.

`Training for more epochs resulted in the Generator loss becoming NaN and breaking the training loop. This same behaviour was observed even earlier in the K80s and P100s provided by colab.`

Thanks to [ganhacks by Soumith Chintala](https://github.com/soumith/ganhacks) and [Tensorflow's DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan) for the resources and insights provided.
