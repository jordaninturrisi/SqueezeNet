from keras.models import Model
from keras.layers import Input, Activation, Concatenate, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout

def fire_module(name,x,sp,e11p,e33p,act,data_format,init):
    """
    Fire module of SqueezeNet model

    Inputs:
    - name: name of fire module
    - x: input to fire module
    - sp: number of filters for 1x1 squeeze
    - e11p: number of filters for 1x1 conv expand
    - e33p: number of filters for 3x3 conv expand
    - act: activation function for squeeze & expand layers
    - data_format: 'channels_first' or 'channels_last'
    - init: kernel initializer

    Returns:
    - output: output of fire module
    """
    squeeze = Conv2D(filters=sp, kernel_size=[1,1], activation=act,
    kernel_initializer=init, padding='same', name=name+'_squeeze',
    use_bias='True', data_format=data_format)(x)

    e11 = Conv2D(filters=e11p, kernel_size=[1,1], activation=act,
    kernel_initializer=init, padding='same', name=name+'_e11',
    use_bias='True', data_format=data_format)(squeeze)

    e33 = Conv2D(filters=e33p, kernel_size=[3,3], activation=act,
    kernel_initializer=init, padding='same', name=name+'_e33',
    use_bias='True', data_format=data_format)(squeeze)

    if (data_format == 'channels_first'):
        output = Concatenate(axis=1)([e11, e33])
    elif (data_format == 'channels_last'):
        output = Concatenate(axis=3)([e11, e33])

    return output


def SqueezeNet_v1_0(H,W,C,num_classes,act):
    """
    Keras implementation of SqueezeNet v1.0 (arXiv: 1602.07360)
    (https://arxiv.org/abs/1602.07360)
    (https://github.com/DeepScale/SqueezeNet)

    Inputs:
    - H: number of vertical pixels
    - W: number of horizontal pixels
    - C: number of input channels (generally number of colour channels)
    - num_classes: total number of final categories
    - act: activation function

    Returns:
    - model: SqueezeNet v1.0 model
    """

    data_format = 'channels_last'
    init = 'glorot_uniform'

    if (data_format == 'channels_first'):
        input_img = Input(shape=[C,H,W])
    elif (data_format == 'channels_last'):
        input_img = Input(shape=[H,W,C])
    else:
        print('ValueError: The `data_format` argument must be one of "channels_first", "channels_last"')

    conv1 = Conv2D(filters=96, kernel_size=[7,7], activation=act,
    kernel_initializer=init, strides=[2,2], padding='same',
    name='conv1', data_format=data_format)(input_img)

    maxpool1 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool1',
    data_format=data_format)(conv1)

    fire2 = fire_module('fire2',maxpool1,16,64,64,act,data_format,init)

    fire3 = fire_module('fire3',fire2,16,64,64,act,data_format,init)

    fire4 = fire_module('fire4',fire3,32,128,128,act,data_format,init)

    maxpool4 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool4',
    data_format=data_format)(fire4)

    fire5 = fire_module('fire5',maxpool4,32,128,128,act,data_format,init)

    fire6 = fire_module('fire6',fire5,48,192,192,act,data_format,init)

    fire7 = fire_module('fire7',fire6,48,192,192,act,data_format,init)

    fire8 = fire_module('fire8',fire7,64,256,256,act,data_format,init)

    maxpool8 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool8',
    data_format=data_format)(fire8)

    fire9 = fire_module('fire9',maxpool8,64,256,256,act,data_format,init)

    fire9_dropout = Dropout(rate=0.5, name='fire9_dropout')(fire9)

    conv10 = Conv2D(filters=num_classes, kernel_size=[1,1],
    kernel_initializer=init, padding='valid', name='conv10',
    data_format=data_format)(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format=data_format)(conv10)
    softmax = Activation(activation="softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax)


def SqueezeNet_v1_1(H,W,C,num_classes,act):
    """
    Keras implementation of SqueezeNet v1.1 (arXiv: 1602.07360)
    (https://arxiv.org/abs/1602.07360)
    (https://github.com/DeepScale/SqueezeNet)

    Inputs:
    - H: number of vertical pixels
    - W: number of horizontal pixels
    - C: number of input channels (generally number of colour channels)
    - num_classes: total number of final categories
    - act: activation function

    Returns:
    - model: SqueezeNet v1.1 model
    """

    data_format = 'channels_last'
    init = 'glorot_uniform'

    if (data_format == 'channels_first'):
        input_img = Input(shape=[C,H,W])
    elif (data_format == 'channels_last'):
        input_img = Input(shape=[H,W,C])
    else:
        print('ValueError: The `data_format` argument must be one of "channels_first", "channels_last"')

    conv1 = Conv2D(filters=64, kernel_size=[3,3], activation=act,
    kernel_initializer=init, strides=[2,2], padding='same',
    name='conv1', data_format=data_format)(input_img)

    maxpool1 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool1',
    data_format=data_format)(conv1)

    fire2 = fire_module('fire2',maxpool1,16,64,64,act,data_format,init)

    fire3 = fire_module('fire3',fire2,16,64,64,act,data_format,init)

    maxpool3 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool3',
    data_format=data_format)(fire3)

    fire4 = fire_module('fire4',maxpool3,32,128,128,act,data_format,init)

    fire5 = fire_module('fire5',fire4,32,128,128,act,data_format,init)

    maxpool5 = MaxPooling2D(pool_size=[3,3], strides=[2,2], name='maxpool5',
    data_format=data_format)(fire5)

    fire6 = fire_module('fire6',maxpool5,48,192,192,act,data_format,init)

    fire7 = fire_module('fire7',fire6,48,192,192,act,data_format,init)

    fire8 = fire_module('fire8',fire7,64,256,256,act,data_format,init)

    fire9 = fire_module('fire9',fire8,64,256,256,act,data_format,init)

    fire9_dropout = Dropout(rate=0.5, name='fire9_dropout')(fire9)

    conv10 = Conv2D(filters=num_classes, kernel_size=[1,1],
    kernel_initializer=init, padding='valid', name='conv10',
    data_format=data_format)(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format=data_format)(conv10)
    softmax = Activation(activation="softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_img, outputs=softmax)
