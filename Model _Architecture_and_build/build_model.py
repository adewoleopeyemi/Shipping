from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class build_nudity_detection_alogrithm:
  
  def __init__(self, X, y, batch_size=7, epochs=10, shuffle=True, learning_rate=0.1, for_deployment=True):
    print(bcolors.WARNING + "Note: Each and every hyper-parameters of this model has been tuned to work optimally\nplease do not change them without the recommendations of an expert" + bcolors.ENDC)
    self.X=X
    self.y=y
    self.batch_size=batch_size
    self.epochs=10
    self.shuffle=shuffle
    self.learning_rate=learning_rate
    self.for_deployment=for_deployment
  
  
  def _small_weighted_model(self, lr):
    '''This function builds a small version of the nudity detection algorithm
    it should only be used for a testing environment

    lr: specifies the learning rate of the model -type:float

    rtype: tf.keras.model
    '''
    small_model = models.Sequential()
    small_model.add(layers.Conv2D(128, 3, activation='relu', input_shape=(124,124,3)))
    small_model.add(layers.Conv2D(128, 3, activation="relu"))
    small_model.add(layers.MaxPooling2D())
    small_model.add(layers.Conv2D(256, 3, activation='relu'))
    small_model.add(layers.Conv2D(256, 3, activation="relu"))
    small_model.add(layers.MaxPooling2D())
    small_model.add(layers.Flatten())
    small_model.add(layers.Dense(1, activation='sigmoid'))
    small_model.compile(loss = binary_crossentropy, optimizer=Adam(lr), metrics=['acc'])
    small_model.summary()
    print(bcolors.WARNING + "Warning: This model is the small version of this algorithm and might\nhave a low accuracy please use the large version for deployment" + bcolors.ENDC)
    return small_model

  def _deployment_model(self,lr):
    """This is a standard neural network initialized with weights learnt from VGG16 model initialized with weights
    learn't from imagenet dataset
    
    """
    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(124, 124, 3))
    deploy_model = models.Sequential()
    deploy_model.add(conv_base)
    deploy_model.add(layers.Flatten())
    deploy_model.add(layers.Dense(1, activation='sigmoid'))
    deploy_model.compile(loss = binary_crossentropy, optimizer=Adam(lr), metrics=['acc'])
    deploy_model.summary()
    return deploy_model

  def train(self, model, X, y, epochs, batch_size ,shuffle):
    #train the model
    model = model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=shuffle)
    return model
  def main(self):
    if self.for_deployment:
      # deployment model
      model = self._deployment_model(self.learning_rate)
    else:
      #testing model
      self.learning_rate = 0.001
      model = self._small_weighted_model(self.learning_rate)
    model = self.train(model, self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, shuffle=self.shuffle)
    model = model.save("Trained_model.h5")