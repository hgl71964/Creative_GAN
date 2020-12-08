import tensorflow as tf
import tensorflow.keras as kr
import os

class Improved_GAN:
    #  use technique from Improved Training of Wasserstein GANs
    def __init__(self, 
                model =(None, None),  # (generator, discriminator)
                **kwargs,  #  for hyper-parameter
                ):
        self.epoch = kwargs["epoch"]
        self.noise_dim = kwargs["noise_dim"]
        self.noise_batch_size = kwargs["noise_batch_size"]
        self.G_opt = kr.optimizers.Adam(kwargs["lr"][0])
        self.D_opt = kr.optimizers.Adam(kwargs["lr"][1])
        self.device = kwargs["device"]
        self.generator = model[0]
        self.discriminator = model[1]

    
    def train(self, image_dataset):
        for epoch in range(self.epoch):
            print("epoch: ", epoch+1)

            total_G_loss, total_D_loss = tf.zeros((1,)), tf.zeros((1,))

            for real_images in image_dataset:
                G_loss, D_loss = self._train_step(real_images)  #  real_images: [batch_size, rows, cols, channels]: (batch_size, 28, 28, channel_nums)
                total_G_loss+=G_loss
                total_D_loss+=D_loss

            tf.print(epoch+1)
            tf.print("G_loss: ", total_G_loss)
            tf.print("D_loss: ", total_D_loss)
            #     self.checkpoint.save(file_prefix = "origin_gan")
                
    @tf.function
    def _train_step(self, real_images):   
        noise = tf.random.normal([self.noise_batch_size, self.noise_dim])

        with tf.device(f"{self.device}"):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # one training step, i.e. the k hyperparameter = 1;
                fake_images = self.generator(noise, training = True)

                # tf.print(fake_images.shape)
                # tf.print(real_images.shape)

                real_output = self.discriminator(real_images, training = True)   #  D(x) -> [batch_size, 1]
                fake_output = self.discriminator(fake_images, training = True)    #  D(G(z)) -> [batch_size, 1]

                G_loss = self._G_loss(fake_output)  #  -E{D[G(z)]}
                D_loss = self._D_loss(real_output, fake_output, G_loss) # E{D[G(z)]} - E{D(x)}
                Grad_penalty = self.gra

            G_grad = gen_tape.gradient(G_loss, self.generator.trainable_variables)
            D_grad = disc_tape.gradient(D_loss, self.discriminator.trainable_variables)

            self.G_opt.apply_gradients(zip(G_grad, self.generator.trainable_variables))
            self.D_opt.apply_gradients(zip(D_grad, self.discriminator.trainable_variables))
        return G_loss, D_loss
    

    def gradient_penalty(self, real_output, fake_output):
        return 

    def _D_loss(self, real_output, G_loss):
        return -G_loss - tf.reduce_mean(real_output)

    def _G_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)