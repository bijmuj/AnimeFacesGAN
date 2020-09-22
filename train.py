import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #nopep8
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
from tensorflow.train import Checkpoint, CheckpointManager
from models import generator, discriminator, generator_loss, discriminator_loss
from utils import load_image, generate_and_save_images, get_data, make_gif
import tensorflow as tf
import numpy as np
import time 
ENCODING_SIZE = 100

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
            '-i', 
            '--imgs',
            type=str,
            default='./data/images/',
            required=False,
            help='Path to image files'
    )
    parser.add_argument(
            '-o',
            '--output',
            type=str,
            default='./data/out/',
            required=False,
            help='Path to output files'
    )
    parser.add_argument(
            '-c',
            '--checkpoint',
            type=str,
            default='./ckpt/',
            required=False,
            help='Path to checkpoints'
    )
    parser.add_argument(
            '-b',
            '--batch',
            type=int,
            default=100,
            required=False,
            help='Batch size'
    )
    parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=50,
            required=False,
            help='Number of epochs for training'
    )
    parser.add_argument(
            '-l',
            '--lr',
            type=float,
            default=2e-4,
            required=False,
            help='Learning rate'
    )
    args = parser.parse_args()
    return args


@tf.function
def train_step(images, gen, disc, gen_opt, disc_opt, batch_size):
    noise = tf.random.normal([batch_size, ENCODING_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)

        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss_real, disc_loss_fake = discriminator_loss(real_output, fake_output)
        disc_loss_total = disc_loss_real + disc_loss_fake

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss_total, disc.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    return gen_loss, disc_loss_real, disc_loss_fake


def train(epochs, batch_size, ckpt_path, imgs_path, lr, out_path):
    tf.keras.backend.clear_session()
    train_data = get_data(imgs_path, batch_size)
    gen = generator()
    disc = discriminator()

    print(gen.summary())
    print(disc.summary())
    gen_opt = Adam(learning_rate=lr, beta_1=0.5)
    disc_opt = Adam(learning_rate=lr, beta_1=0.5)

    ckpt = Checkpoint(disc=disc, gen=gen, 
            disc_opt=disc_opt, gen_opt=gen_opt)
    manager = CheckpointManager(ckpt, ckpt_path, max_to_keep=3)
    
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        ckpt.restore(manager.latest_checkpoint)
    else:
        print("Initializing from scratch.")

    seed = tf.random.normal([16, ENCODING_SIZE], seed=1234)

    generate_and_save_images(gen, 0, seed, out_path)
    for ep in range(epochs):
        gen_loss = []
        disc_loss_real = []
        disc_loss_fake = []
        print('Epoch: %d of %d'%(ep + 1, epochs))
        start = time.time()

        for images in train_data:
            g_loss, d_loss_r, d_loss_f = train_step(images, gen, disc, gen_opt, disc_opt, batch_size)
            gen_loss.append(g_loss)
            disc_loss_real.append(d_loss_r)
            disc_loss_fake.append(d_loss_f)
        gen_loss = np.mean(np.asarray(gen_loss))
        disc_loss_real = np.mean(np.asarray(disc_loss_real))
        disc_loss_fake = np.mean(np.asarray(disc_loss_fake))

        if (np.isnan(gen_loss) or np.isnan(disc_loss_real) or np.isnan(disc_loss_fake)):
            print("Something broke.")
            break
        
        manager.save()
        generate_and_save_images(gen, ep + 1, seed, out_path)

        print("Time for epoch:", time.time()-start)
        print("Gen loss=", gen_loss)
        print("Disc loss real=", disc_loss_real)
        print("Disc loss fake=", disc_loss_fake)


if __name__ == "__main__":
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch
    ckpt_path = args.checkpoint
    imgs_path = args.imgs
    out_path = args.output
    lr = args.lr
    
    train(epochs, batch_size, ckpt_path, imgs_path, lr, out_path)
    make_gif(out_path)