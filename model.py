from utils import getSets
from utils import generator

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import Convolution1D
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Dense, TimeDistributed
from keras.layers import GlobalMaxPool1D, MaxPool1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

modelPath = '/content/drive/My Drive/sleepStageData/cnnFinal.h5'
trainDict, testDict, valDict = getSets()

def epochEncoder():
    """
    This is a sub-model that encodes each 30-second EEG epoch into a 1D vector.
    A 1D CNN is used to encode the epochs.
    """
    # we are dealing with time series data
    inputShape = Input(shape=(3000, 1))

    x = Convolution1D(16, kernel_size=5, activation="relu", padding="valid")(inputShape)
    x = Convolution1D(16, kernel_size=5, activation="relu", padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.01)(x)

    x = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(x)
    x = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.01)(x)

    x = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(x)
    x = Convolution1D(32, kernel_size=3, activation="relu", padding="valid")(x)
    x = MaxPool1D(pool_size=2)(x)
    x = SpatialDropout1D(rate=0.01)(x)

    x = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(x)
    x = Convolution1D(256, kernel_size=3, activation="relu", padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.01)(x)

    denseOne = Dropout(0.01)(Dense(64, activation="relu", name="denseOne")(x))

    epochEncoder = Model(inputs=inputShape, outputs=denseOne)
    optAdam = Adam(0.001)

    epochEncoder.compile(optimizer=optAdam, loss="sparse_categorical_crossentropy", metrics=['acc'])

    return epochEncoder

def epochLabeller(existingModel):
    """
    This is a sub-model that labels a sequence of EEG epochs encoded the by the first model.
    A 1D CNN is used to label the epochs.
    """
    # classify epoch as either "W", "N1", "N2", "N3", or "REM"
    numClass = 5

    seqInput = Input(shape=(None, 3000, 1))

    # apply the first sub-model to each EEG epoch
    encodedSequence = TimeDistributed(existingModel)(seqInput)

    encodedSequence = SpatialDropout1D(rate=0.01)(Convolution1D(128, kernel_size=3, activation="relu", padding="same")(encodedSequence))
    encodedSequence = Dropout(rate=0.05)(Convolution1D(128, kernel_size=3, activation="relu", padding="same")(encodedSequence))

    out = Convolution1D(numClass, kernel_size=3, activation="softmax", padding="same")(encodedSequence)

    epochLabeller = Model(seqInput, out)
    epochLabeller.compile(Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])

    return epochLabeller

# stack the two sub-models
epochEncoder = epochEncoder()
model = epochLabeller(epochEncoder)

model.summary()

# use the validation accuracy as a monitor for callbacks
checkPoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlyStop = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
reduceOnPlat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)

callbacksList = [checkPoint, earlyStop, reduceOnPlat]

# train and save the model
model.fit_generator(generator(trainDict, aug=False), validation_data=generator(valDict), epochs=100, verbose=2, steps_per_epoch=1000, validation_steps=300, callbacks=callbacksList)
model.save(modelPath)