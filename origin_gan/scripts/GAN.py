"""
GAN implementation
"""
import tensorflow as tf
import tensorflow.keras as kr
import generator as generator
import discriminator as discriminator
import os

class GAN:

    def __init__(self, 
                batch_size, #  int, 
                epoch,  #  int,
                noise_dim,  #  int,
                lr = (1e-4, 1e-4),  #  learning rate,  tuple -> [generator_lr, discriminator_lr]
                checkpoint_prefix = "origin_gan",  #  string
                ):
        self.epoch = epoch
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.G_opt = kr.optimizers.Adam(lr[0])
        self.D_opt = kr.optimizers.Adam(lr[1])  

        self.generator = generator.generator_model()
        self.discriminator = discriminator.discriminator_model() 

        self.loss_metric = kr.losses.BinaryCrossentropy()  #   from_logits=True -> smoother? 

        self.checkpoint(checkpoint_prefix = checkpoint_prefix)

    
    def train(self, image_dataset):

        for epoch in range(self.epoch):
            for real_images in image_dataset:
                self._train_step(real_images)  #  real_images: [batch_size, rows, cols, channels]: (batch_size, 28, 28, 1)

            if (epoch + 1) % 15 == 0:   # output stats && save models
                self.checkpoint.save(file_prefix = "origin_gan")
                
    
    @tf.function
    def _train_step(self, real_images):   # one training step
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training = True)

            real_output = self.discriminator(real_images, training = True)   #  D(x) -> [batch_size, 1]
            fake_output = self.discriminator(fake_images, training = True)    #  D(G(z)) -> [batch_size, 1]

            G_loss = self._G_loss(fake_output)
            D_loss = self._D_loss(real_output, fake_output)

        G_grad = gen_tape.gradient(G_loss, self.generator.trainable_variables)
        D_grad = disc_tape.gradient(D_loss, self.discriminator.trainable_variables)

        self.G_opt.apply_gradients(zip(G_grad, self.generator.trainable_variables))
        self.D_opt.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))

    def _D_loss(self, real_output, fake_output):
        return self.loss_metric(tf.ones_like(real_output), real_output) \
                + self.loss_metric(tf.zeros_like(fake_output), fake_output)  #  -( ylog(p) + (1-y) log (1-p))

    def _G_loss(self, fake_output):
        return self.loss_metric(tf.ones_like(fake_output), fake_output)  #  using the training trick, obtain strong early gradients

    def checkpoint(self, checkpoint_prefix):            
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.G_opt,
                                 discriminator_optimizer=self.D_opt,
                                 generator=self.generator,
                                 discriminator=self.discriminator)


if __name__ == "__main__":
    gan = GAN(batch_size=2, epoch=2, noise_dim=10)


    image_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal(shape=(4, 28, 28, 1), stddev=10)).shuffle(buffer_size=10).batch(2)

    # for i in image_dataset:
    #     print(i.shape)

    gan.train(image_dataset)

