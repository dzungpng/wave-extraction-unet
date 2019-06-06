from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import model

NO_OF_TRAINING_IMAGES = len(os.listdir('/your_data/train_frames/train/'))
NO_OF_VAL_IMAGES = len(os.listdir('/your_data/val_frames/val/'))

NO_OF_EPOCHS = 'ANYTHING FROM 30-100 FOR SMALL-MEDIUM SIZED DATASETS IS OKAY'

BATCH_SIZE = 'BATCH SIZE PREVIOUSLY INITIALISED'

weights_path = 'path/where/resulting_weights_will_be_saved'

m = model.FCN_Vgg16_32s()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

m.compile(loss='The loss to optimise [eg: dice_loss],
              optimizer=opt,
              metrics='YOUR_METRIC [could be 'accuracy' or mIOU, dice_coeff etc]')

checkpoint = ModelCheckpoint(weights_path, monitor='METRIC_TO_MONITOR', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = m.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
m.save('Model.h5')