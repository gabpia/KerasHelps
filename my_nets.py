from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam

from KerasHelps.my_metrics import dice_coef, dice_coef_loss, jaccard_coef


def unet_2d(input_size, learning_rate=1e-5, lr_decay_rate=0., batch_norm=False, dropout=False, dropout_rate=0.25):
    return unet_2d_5layers(input_size, learning_rate, lr_decay_rate, batch_norm, dropout, dropout_rate)


def unet_2d_5layers(input_size, learning_rate=1e-5, lr_decay_rate=0., batch_norm=False, dropout=False, dropout_rate=0.25):
    inputs = Input((input_size[0], input_size[1], 1))
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    if batch_norm:
        conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_1)
    if batch_norm:
        conv1_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    if batch_norm:
        conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_2)
    if batch_norm:
        conv1_2 = BatchNormalization()(conv1_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    if batch_norm:
        conv1_3 = BatchNormalization()(conv1_3)
    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1_3)
    if batch_norm:
        conv1_3 = BatchNormalization()(conv1_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    conv1_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    if batch_norm:
        conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv1_4)
    if batch_norm:
        conv1_4 = BatchNormalization()(conv1_4)
    if dropout:
        conv1_4 = Dropout(dropout_rate)(conv1_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv1_4)

    conv1_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    if batch_norm:
        conv1_5 = BatchNormalization()(conv1_5)
    conv1_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv1_5)
    if batch_norm:
        conv1_5 = BatchNormalization()(conv1_5)
    if dropout:
        conv1_5 = Dropout(dropout_rate)(conv1_5)

    # up1 = UpSampling2D(size=[2,2],conv1_5);
    up1 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv1_5), conv1_4], axis=3)
    conv2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    if batch_norm:
        conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2_1)
    if batch_norm:
        conv2_1 = BatchNormalization()(conv2_1)

    up2 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_1), conv1_3], axis=3)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    if batch_norm:
        conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_2)
    if batch_norm:
        conv2_2 = BatchNormalization()(conv2_2)

    up3 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2_2), conv1_2], axis=3)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    if batch_norm:
        conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_3)
    if batch_norm:
        conv2_3 = BatchNormalization()(conv2_3)

    up4 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3), conv1_1], axis=3)
    conv2_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
    if batch_norm:
        conv2_4 = BatchNormalization()(conv2_4)
    conv2_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_4)
    if batch_norm:
        conv2_4 = BatchNormalization()(conv2_4)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv2_4)

    # MODEL
    model = Model(inputs=[inputs], outputs=[conv10])

    # OPTIMIZER
    opt = Adam(lr=learning_rate, decay=lr_decay_rate)

    # COMPILE
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef, jaccard_coef, "accuracy"])

    return model


def unet_2d_6layers(input_size, learning_rate=1e-5, lr_decay_rate=0., batch_norm=False, dropout=False, dropout_rate=0.25):
    inputs = Input((input_size[0], input_size[1], 1))
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    if batch_norm:
        conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_1)
    if batch_norm:
        conv1_1 = BatchNormalization()(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    if batch_norm:
        conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_2)
    if batch_norm:
        conv1_2 = BatchNormalization()(conv1_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    if batch_norm:
        conv1_3 = BatchNormalization()(conv1_3)
    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1_3)
    if batch_norm:
        conv1_3 = BatchNormalization()(conv1_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    conv1_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    if batch_norm:
        conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv1_4)
    if batch_norm:
        conv1_4 = BatchNormalization()(conv1_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv1_4)

    conv1_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    if batch_norm:
        conv1_5 = BatchNormalization()(conv1_5)
    conv1_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv1_5)
    if batch_norm:
        conv1_5 = BatchNormalization()(conv1_5)
    if dropout:
        conv1_5 = Dropout(dropout_rate)(conv1_5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv1_5)

    conv1_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool5)
    if batch_norm:
        conv1_6 = BatchNormalization()(conv1_6)
    conv1_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv1_6)
    if batch_norm:
        conv1_6 = BatchNormalization()(conv1_6)
    if dropout:
        conv1_6 = Dropout(dropout_rate)(conv1_6)

    # up1 = UpSampling2D(size=[2,2],conv1_5);
    up0 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv1_6), conv1_5], axis=3)
    conv2_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(up0)
    if batch_norm:
        conv2_0 = BatchNormalization()(conv2_0)
    conv2_0 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv2_0)
    if batch_norm:
        conv2_0 = BatchNormalization()(conv2_0)

    up1 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv2_0), conv1_4], axis=3)
    conv2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    if batch_norm:
        conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2_1)
    if batch_norm:
        conv2_1 = BatchNormalization()(conv2_1)

    up2 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_1), conv1_3], axis=3)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    if batch_norm:
        conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_2)
    if batch_norm:
        conv2_2 = BatchNormalization()(conv2_2)

    up3 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv2_2), conv1_2], axis=3)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    if batch_norm:
        conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2_3)
    if batch_norm:
        conv2_3 = BatchNormalization()(conv2_3)

    up4 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3), conv1_1], axis=3)
    conv2_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
    if batch_norm:
        conv2_4 = BatchNormalization()(conv2_4)
    conv2_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_4)
    if batch_norm:
        conv2_4 = BatchNormalization()(conv2_4)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv2_4)

    # MODEL
    model = Model(inputs=[inputs], outputs=[conv10])

    # OPTIMIZER
    opt = Adam(lr=learning_rate, decay=lr_decay_rate)

    # COMPILE
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef, jaccard_coef, "accuracy"])

    return model


def autoencoder_2d(input_size, depth):
    encoding_layers = []
    decoding_layers = []

    # ENCODING
    encoding_layers.append(Input((input_size[0], input_size[1], 1), name='input'))
    for i in range(depth):
        encoding_layers.append(Conv2D(8, (3, 3), activation='relu', padding='same', name='en_conv_%d' % i)(encoding_layers[-1]))
        encoding_layers.append(MaxPooling2D((2, 2), padding='same', name='en_pool_%d' % i)(encoding_layers[-1]))

    # DECODING
    decoding_layers.append(encoding_layers[-1])
    for i in range(depth):
        decoding_layers.append(Conv2D(8, (3, 3), activation='relu', padding='same', name='de_conv_%d' % i)(decoding_layers[-1]))
        decoding_layers.append(UpSampling2D((2, 2), name='de_upsample_%d' % i)(decoding_layers[-1]))
    decoding_layers.append(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output')(decoding_layers[-1]))

    model = Model(encoding_layers[0], decoding_layers[-1])
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    return model
