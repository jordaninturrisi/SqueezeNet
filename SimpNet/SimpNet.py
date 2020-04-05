import keras

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Dense,
    Activation,
    Flatten,
    Dropout,
    BatchNormalization,
)
from keras.initializers import glorot_normal, RandomNormal, Zeros


def block(name, x, F, K, act, init, pool=False):
    """
    Define block which is a homogenous grouping of layers used to build SimpNet.
    """
    out = Conv2D(
        filters=F,
        kernel_size=K,
        strides=1,
        use_bias=True,
        padding="same",
        kernel_initializer=init,
        name=name,
    )(x)
    out = BatchNormalization(
        momentum=0.95, center=True, scale=True, name=name + "_batchnorm"
    )(out)
    out = Activation(act, name=name + "_act")(out)

    if pool == "Max":
        out = MaxPooling2D(pool_size=2, strides=2, name=name + "_maxpool")(out)
    elif pool == "Global":
        out = GlobalMaxPooling2D(name=name + "_globalpool")(out)

    out = Dropout(0.2, name=name + "_dropout")(out)

    return out


def SimpNet_CIFAR10_9M(H, W, C, num_classes, act, drop_rate=0.25, reg=0):
    """
    Keras implementation of SimpNet (arXiv: 1802.06205)
    (https://arxiv.org/abs/1802.06205)

    SimpNet model for CIFAR10/100 with 8.9M parameters

    Inputs:
    - H: number of vertical pixels
    - W: number of horizontal pixels
    - C: number of input channels (generally number of colour channels)
    - num_classes: total number of final categories

    Returns:
    - model: SimpNet model
    """

    input_img = Input(shape=[H, W, C], name="Input")

    # Block 1
    block1 = Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        use_bias=True,
        padding="same",
        kernel_initializer=glorot_normal(),
        input_shape=[H, W, C],
        name="block1",
    )(input_img)
    block1 = BatchNormalization(
        momentum=0.95, center=True, scale=True, name="block1_batchnorm"
    )(block1)
    block1 = Activation(act, name="block1_act")(block1)
    block1 = Dropout(0.2, name="block1_dropout")(block1)

    # Block 2
    block2 = block("block2", block1, 182, 3, act, glorot_normal())

    # Block 3
    block3 = block("block3", block2, 182, 3, act, RandomNormal(stddev=0.01))

    # Block 4
    block4 = block("block4", block3, 182, 3, act, RandomNormal(stddev=0.01))

    # Block 5
    block5 = block("block5", block4, 182, 3, act, RandomNormal(stddev=0.01), pool="Max")

    # Block 6
    block6 = block("block6", block5, 182, 3, act, glorot_normal())

    # Block 7
    block7 = block("block7", block6, 182, 3, act, glorot_normal())

    # Block 8
    block8 = block("block8", block7, 182, 3, act, glorot_normal())

    # Block 9
    block9 = block("block9", block8, 182, 3, act, glorot_normal())

    # Block 10
    block10 = block("block10", block9, 430, 3, act, glorot_normal(), pool="Max")

    # Block 11
    block11 = block("block11", block10, 430, 3, act, glorot_normal())

    # Block 12
    block12 = block("block12", block11, 455, 3, act, glorot_normal())

    # Block 13
    block13 = block("block13", block12, 600, 3, act, glorot_normal(), pool="Global")

    # Final Classifier
    softmax = Dense(
        num_classes,
        activation="softmax",
        name="softmax",
        kernel_initializer=glorot_normal(),
        bias_initializer=Zeros(),
    )(block13)

    return Model(inputs=input_img, outputs=softmax)


def SimpNet_CIFAR10_5M(H, W, C, num_classes, act, drop_rate=0.25, reg=0):
    """
    Keras implementation of SimpNet (arXiv: 1802.06205)
    (https://arxiv.org/abs/1802.06205)

    SimpNet model for CIFAR10/100 with 5M parameters
    Also SimpNet model for SVHN (Street View House Numbers)

    Inputs:
    - H: number of vertical pixels
    - W: number of horizontal pixels
    - C: number of input channels (generally number of colour channels)
    - num_classes: total number of final categories

    Returns:
    - model: SimpNet model
    """

    input_img = Input(shape=[H, W, C], name="Input")

    # Block 1
    block1 = Conv2D(
        filters=66,
        kernel_size=3,
        strides=1,
        use_bias=True,
        padding="same",
        kernel_initializer=glorot_normal(),
        input_shape=[H, W, C],
        name="block1",
    )(input_img)
    block1 = BatchNormalization(
        momentum=0.95, center=True, scale=True, name="block1_batchnorm"
    )(block1)
    block1 = Activation(act, name="block1_act")(block1)
    block1 = Dropout(0.2, name="block1_dropout")(block1)

    # Block 2
    block2 = block("block2", block1, 128, 3, act, glorot_normal())

    # Block 3
    block3 = block("block3", block2, 128, 3, act, RandomNormal(stddev=0.01))

    # Block 4
    block4 = block("block4", block3, 128, 3, act, RandomNormal(stddev=0.01))

    # Block 5
    block5 = block("block5", block4, 192, 3, act, RandomNormal(stddev=0.01), pool="Max")

    # Block 6
    block6 = block("block6", block5, 192, 3, act, glorot_normal())

    # Block 7
    block7 = block("block7", block6, 192, 3, act, glorot_normal())

    # Block 8
    block8 = block("block8", block7, 192, 3, act, glorot_normal())

    # Block 9
    block9 = block("block9", block8, 192, 3, act, glorot_normal())

    # Block 10
    block10 = block("block10", block9, 288, 3, act, glorot_normal(), pool="Max")

    # Block 11
    block11 = block("block11", block10, 288, 3, act, glorot_normal())

    # Block 12
    block12 = block("block12", block11, 355, 3, act, glorot_normal())

    # Block 13
    block13 = block("block13", block12, 432, 3, act, glorot_normal(), pool="Global")

    # Final Classifier
    softmax = Dense(
        num_classes,
        activation="softmax",
        name="softmax",
        kernel_initializer=glorot_normal(),
        bias_initializer=Zeros(),
    )(block13)

    return Model(inputs=input_img, outputs=softmax)


def SimpNet_MNIST(H, W, C, num_classes, act, drop_rate=0.25, reg=0):
    """
    Keras implementation of SimpNet (arXiv: 1802.06205)
    (https://arxiv.org/abs/1802.06205)

    SimpNet model for MNIST.

    Inputs:
    - H: number of vertical pixels
    - W: number of horizontal pixels
    - C: number of input channels (generally number of colour channels)
    - num_classes: total number of final categories

    Returns:
    - model: SimpNet model
    """

    input_img = Input(shape=[H, W, C], name="Input")

    # Block 1
    block1 = Conv2D(
        filters=66,
        kernel_size=3,
        strides=1,
        use_bias=True,
        padding="same",
        kernel_initializer=glorot_normal(),
        input_shape=[H, W, C],
        name="block1",
    )(input_img)
    block1 = BatchNormalization(
        momentum=0.95, center=True, scale=True, name="block1_batchnorm"
    )(block1)
    block1 = Activation(act, name="block1_act")(block1)
    block1 = Dropout(0.2, name="block1_dropout")(block1)

    # Block 2
    block2 = block("block2", block1, 64, 3, act, glorot_normal())

    # Block 3
    block3 = block("block3", block2, 64, 3, act, RandomNormal(stddev=0.01))

    # Block 4
    block4 = block("block4", block3, 64, 3, act, RandomNormal(stddev=0.01))

    # Block 5
    block5 = block("block5", block4, 96, 3, act, RandomNormal(stddev=0.01), pool="Max")

    # Block 6
    block6 = block("block6", block5, 96, 3, act, glorot_normal())

    # Block 7
    block7 = block("block7", block6, 96, 3, act, glorot_normal())

    # Block 8
    block8 = block("block8", block7, 96, 3, act, glorot_normal())

    # Block 9
    block9 = block("block9", block8, 96, 3, act, glorot_normal())

    # Block 10
    block10 = block("block10", block9, 144, 3, act, glorot_normal(), pool="Max")

    # Block 11
    block11 = block("block11", block10, 144, 3, act, glorot_normal())

    # Block 12
    block12 = block("block12", block11, 178, 3, act, glorot_normal())

    # Block 13
    block13 = block("block13", block12, 216, 3, act, glorot_normal(), pool="Global")

    # Final Classifier
    softmax = Dense(
        num_classes,
        activation="softmax",
        name="softmax",
        kernel_initializer=glorot_normal(),
        bias_initializer=Zeros(),
    )(block13)

    return Model(inputs=input_img, outputs=softmax)
