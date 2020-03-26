"""
Networks implementation for

Krebs, Julian, Hervé Delingette, Boris Mailhé, Nicholas Ayache, and Tommaso Mansi.
"Learning a probabilistic model for diffeomorphic registration."
IEEE transactions on medical imaging 38, no. 9 (2019): 2165-2176.
"""
# main imports
import sys
# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import Reshape, Dense
from keras.layers import LeakyReLU, Reshape
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils

# other vm functions
import losses

def cvae(vol_size, int_steps=7, use_miccai_int=False, indexing='ij', bidir=False):
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # inputs
    # vol_size = 160*128*128
    src = Input(shape=vol_size + (1,))   #[64,64,64,1]
    src2 = trf_resize(src, 2, name='src2')#[32,32,32,1]
    src3 = trf_resize(src2, 2, name='src3')#[16,16,16,1]
    tgt1 = Input(shape=vol_size + (1,))  #[64,64,64,1]
#    tgt2 = trf_resize(tgt1, 2, name='tgt2')#[32,32,32,1]
#    tgt3 = trf_resize(tgt2, 2, name='tgt3')#[16,16,16,1]
    x_in = concatenate([src, tgt1])


    ####### down-sample path (encoder)
    Conv = getattr(KL, 'Conv%dD' % ndims)
    x = Conv(16, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=2)(x_in) #[32,32,32]
    x = LeakyReLU(0.2)(x)
    x = Conv(32, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=2)(x)  #[16,16,16]
    x = LeakyReLU(0.2)(x)
    x = Conv(32, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=2)(x)  #[8,8,8]
    x = LeakyReLU(0.2)(x)
    x = Conv(4, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(x)  #[8,8,8]
    x = LeakyReLU(0.2)(x)

    ####### latent variables / bottle neck layer
    x = Reshape((1,-1))(x)
    # velocity mean and logsigma layers
    flow_mean = Dense(32, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_mean')(x)
    # we're going to initialize the velocity variance very low, to start stable.
    flow_log_sigma = Dense(32, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow_sigma')(x)
    flow_params = concatenate([flow_mean, flow_log_sigma])
    # velocity sample
    f = Sample(name="z_sample")([flow_mean, flow_log_sigma])
    f = Dense(1024, name='fc')(f) #16*1024
    f = Reshape((16,16,-1))(f)#[40,32,32,-1]             [16,16,16,10]

    ####### up-sample path (decoder)
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    f_coarse = concatenate([f, src3]) #[16,16,16,5]
    f = Conv(32, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(f_coarse)
    f = LeakyReLU(0.2)(f)
    f = upsample_layer()(f)

    f_middle = concatenate([f, src2])
    f = Conv(32, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(f_middle)
    f = LeakyReLU(0.2)(f)
    f = upsample_layer()(f)

    f = concatenate([f, src])
    f = Conv(32, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=1)(f)
    f = LeakyReLU(0.2)(f)

    ####### after cvae
    ## full scale
    f = conv_block(f, 16) #16
    f = conv_block(f, 2) #3
    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    z_sample = f
    flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(z_sample)
    # transform
    y_full = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    # prepare outputs and losses
    outputs = [y_full,flow_params]

    # build the model
    return Model(inputs=[src,tgt1], outputs=outputs)


# Helper functions
def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out



def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z

def trf_resize(trf, vel_resize, name='flow'):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return  nrn_layers.Resize(1/vel_resize, name=name)(trf)


class Sample(Layer):
    """
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Negate(Layer):
    """
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape

class Rescale(Layer):
    """
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize

    def compute_output_shape(self, input_shape):
        return input_shape

class RescaleDouble(Rescale):
    def __init__(self, **kwargs):
        self.resize = 2
        super(RescaleDouble, self).__init__(self.resize, **kwargs)

class ResizeDouble(nrn_layers.Resize):
    def __init__(self, **kwargs):
        self.zoom_factor = 2
        super(ResizeDouble, self).__init__(self.zoom_factor, **kwargs)
