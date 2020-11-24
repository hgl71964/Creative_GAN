"""
GAN implementation
"""
import tensorflow as tf
import tensorflow.keras as kr
import generator as generator
import discriminator as discriminator

class GAN:

    def __init__(self, 
                batch_size, #  int, 
                epoch,  #  int,
                noise_dim,  #  tuple, [batchsize, noise_dim]
                lr = (1e-4, 1e-4),  #  learning rate,  tuple -> [generator_lr, discriminator_lr]
                ):
        self.epoch = epoch
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.generator_optimizer = kr.optimizers.Adam(lr[0])
        self.discriminator_optimizer = kr.optimizers.Adam(lr[1])  

        self.generator = generator.generator_model()
        self.discriminator = discriminator.discriminator_model() 

        self.loss_metric = kr.losses.BinaryCrossentropy()  #   from_logits=True -> smoother? 

    
    def train(self, real_images):

        for epoch in range(self.epochs):
            pass

        return 

    
    @tf.function
    def train_step(self, real_images):   # one training step
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training = True)

            real_output = self.discriminator(real_images, training = True)   #  D(x) -> [batch_size, 1]
            fake_output = self.discriminator(fake_images, training = True)    #  D(G(z)) -> [batch_size, 1]

            gen_loss = self.G_loss(fake_output)
            disc_loss = self.D_loss(real_output, fake_output)

        G_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        D_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(G_grad, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(D_grad, discriminator.trainable_variables))

    def D_loss(self, real_output, fake_output):
        return self.loss_metric(tf.ones_like(real_output), real_output) \
                + self.loss_metric(tf.zeros_like(fake_output), fake_output)  #  -( ylog(p) + (1-y) log (1-p))

    def G_loss(self, fake_output):
        return self.loss_metric(tf.ones_like(fake_output), fake_output)  #  the training trick, obtain strong early gradients