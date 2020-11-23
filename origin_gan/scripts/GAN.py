"""
GAN implementation
"""
import tensorflow as tf
import tensorflow.keras as kr
import origin_gan.scripts.generator as generator
import origin_gan.scripts.discriminator as discriminator

class GAN:

    def __init__(self, batch_size, noise_dim):
        super().__init__()

        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.generator_optimizer = kr.optimizers.Adam(1e-4)
        self.discriminator_optimizer = kr.optimizers.Adam(1e-4)    

        self.generator = generator.generator_model()
        self.discriminator = discriminator.discriminator_model()


    
    def train_step(self):
    


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise)

            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_images)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



