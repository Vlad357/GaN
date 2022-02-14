import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import PIL
autotune = tf.data.experimental.AUTOTUNE

monet_filenames = tf.io.gfile.glob('monet_tfrec/*.tfrec')
photo_filenames = tf.io.gfile.glob('photo_tfrec/*.tfrec')

IMAGE_SIZE = [256, 256]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels = 3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape( image, [*IMAGE_SIZE, 3])
    return image

def tfrec_to_image(tfrec_file):
    tfrec_encode = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }
    encoders = tf.io.parse_single_example(tfrec_file, tfrec_encode)
    image = decode_image(encoders['image'])
    return image

def load_dataset(filenames):
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(tfrec_to_image, num_parallel_calls = autotune)
    return ds

monet_ds = load_dataset(monet_filenames).batch(1)
photo_ds = load_dataset(photo_filenames).batch(1)

_, ax = plt.subplots(5, 1, figsize = (12, 12))
for i, img in enumerate(photo_ds.take(5)):
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    
    ax[i].imshow(img)
    ax[i].set_title('Input photo')
    ax[i].axis('off')
plt.show()

monet = next(iter(monet_ds))
photo = next(iter(photo_ds))

OUTPUT_CHANNELS = 3
def downsample(filters, kernel_size, apply_instancenorm = True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2D(filters, kernel_size, strides = 2, padding = 'same', 
                             kernel_initializer = initializer, use_bias = False))
    
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer = gamma_init))
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, kernel_size, apply_dropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)
    
    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, kernel_size, strides = 2, 
                                      padding = 'same', 
                                      kernel_initializer = initializer,
                                      use_bias = False))
    
    result.add(tfa.layers.InstanceNormalization(gamma_initializer = gamma_init))
    
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    inputs = layers.Input(shape = (256, 256, 3))
    
    
    down_stack = [
        downsample(64, 4, apply_instancenorm = False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]
    
    up_stack = [
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4, apply_dropout = True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, 
                                  strides = 2, 
                                  padding = 'same', 
                                  kernel_initializer = initializer, 
                                  activation = 'tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    x = last(x)
    
    return keras.Model(inputs = inputs, outputs = x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)
    
    inp = layers.Input(shape = [256, 256, 3], name = 'input_image')
    
    x = inp
    
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    
    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides = 1, 
                         kernel_initializer = initializer, 
                         use_bias = False)(zero_pad1)
    norm1 = tfa.layers.InstanceNormalization(gamma_initializer = gamma_init)(conv)
    leaky_relu = layers.LeakyReLU()(norm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides = 1, 
                         kernel_initializer = initializer)(zero_pad2)
    return tf.keras.Model(inputs = inp, outputs = last)

monet_generator = Generator()
photo_generator = Generator()

monet_discriminator = Discriminator()
photo_discriminator = Discriminator()

new_monet = monet_generator(monet)

plt.subplot(1, 2, 1)
plt.imshow(monet[0] * 0.5 + 0.5)

plt.subplot(1, 2, 2)
plt.imshow(new_monet[0] * 0.5 + 0.5)

class GaN(keras.Model):
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator, lambda_cycle = 10):
        super(GaN, self).__init__()
        self.monet_gen = monet_generator
        self.photo_gen = photo_generator
        self.monet_disc = monet_discriminator
        self.photo_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
    def compile(self, m_gen_opt, p_gen_opt, m_disc_opt, p_disc_opt, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(GaN, self).compile()
        self.m_gen_opt = m_gen_opt
        self.p_gen_opt = p_gen_opt
        self.m_disc_opt = m_disc_opt
        self.p_disc_opt = p_disc_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        with tf.GradientTape(persistent = True) as tape:
            fake_monet = self.monet_gen(real_photo, training = True)
            cycled_photo = self.photo_gen(fake_monet, training = True)
            
            fake_photo = self.photo_gen(real_monet, training = True)
            cycled_monet = self.monet_gen(fake_photo, training = True)
            
            same_monet = self.monet_gen(real_monet, training = True)
            same_photo = self.photo_gen(real_photo, training = True)
            
            disc_real_monet = self.monet_disc(real_monet, training = True)
            disc_real_photo = self.photo_disc(real_photo, training = True)
            
            disc_fake_monet = self.monet_disc(fake_monet, training = True)
            disc_fake_photo = self.photo_disc(fake_photo, training = True)
            
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)
            
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)
        
        monet_generator_gradients = tape.gradient(total_monet_gen_loss, 
                                                  self.monet_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss, 
                                                  self.photo_gen.trainable_variables)
        monet_discriminator_gradients = tape.gradient(monet_disc_loss, 
                                                      self.monet_disc.trainable_variables)
        photo_disctiminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.photo_disc.trainable_variables)
        self.m_gen_opt.apply_gradients(zip(monet_generator_gradients, 
                                           self.monet_gen.trainable_variables))
        self.p_gen_opt.apply_gradients(zip(photo_generator_gradients, 
                                           self.photo_gen.trainable_variables))
        self.m_disc_opt.apply_gradients(zip(monet_discriminator_gradients, 
                                            self.monet_disc.trainable_variables))
        self.p_disc_opt.apply_gradients(zip(photo_disctiminator_gradients, 
                                            self.photo_disc.trainable_variables))
        return{
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
   
def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True, reduction = tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)   
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

def cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1
def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

monet_gen_opt = keras.optimizers.Adam(2e-4, beta_1 = 0.5)
photo_gen_opt = keras.optimizers.Adam(2e-4, beta_1 = 0.5)

monet_disc_opt = keras.optimizers.Adam(2e-4, beta_1 = 0.5)
photo_disc_opt = keras.optimizers.Adam(2e-4, beta_1 = 0.5)

model = GaN(monet_generator, photo_generator, monet_discriminator, photo_discriminator)
model.compile(
    m_gen_opt = monet_gen_opt,
    p_gen_opt = photo_gen_opt,
    m_disc_opt = monet_disc_opt,
    p_disc_opt = photo_disc_opt,
    gen_loss_fn = generator_loss, 
    disc_loss_fn = discriminator_loss,
    cycle_loss_fn = cycle_loss,
    identity_loss_fn = identity_loss
)

monet_generator = load_model('models/m_generator.h5')
photo_generator = load_model('models/p_generator.h5')

#model.fit(
#    tf.data.Dataset.zip((monet_ds, photo_ds)),
#    epochs = 10
#)

#monet_generator.save('models/m_generator.h5')
#photo_generator.save('models/p_generator.h5')


_, ax = plt.subplots(5, 2, figsize = (12, 12))
for i, img in enumerate(photo_ds.take(5)):
    prediction = monet_generator(img, training = False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    
    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title('Input photo')
    ax[i, 1].set_title('Monet generated')
    ax[i, 0].axis('off')
    ax[i, 1].axis('off')
plt.show()


for i, img in enumerate(photo_ds):
	if i <= 300:
		prediction = monet_generator(img, training = False)[0].numpy()
		prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
		im = PIL.Image.fromarray(prediction)
		im.save('images/' + str(i) + '.jpg')
	else:
		break