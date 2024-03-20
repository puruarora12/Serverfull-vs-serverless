import time


def create_mnist_model():
  from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
  from keras.models import Sequential
  from keras.optimizers import Adam
  from keras.losses import CategoricalCrossentropy
  from keras.metrics import CategoricalAccuracy
  cnn = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'),
    MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax'),
    ])
  cnn.compile(optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossentropy(),
            metrics=["acc"])
  return cnn

def process_mnist():
  import mnist
  from keras.utils import to_categorical
  xtrain, xtest, ytrain, ytest = mnist.train_images()/255., mnist.test_images()/255., mnist.train_labels(), mnist.test_labels()
  ytrain=to_categorical(ytrain)
  ytest=to_categorical(ytest)
  return xtrain,xtest,ytrain,ytest

def train_mnist(model, xtrain,ytrain):
  history = model.fit(xtrain, ytrain, epochs=9, batch_size=256, validation_data=(xtest, ytest))
  return history, model

def evaluate_mnist(model, x_test ,y_test):
  from keras.utils import to_categorical
  y_test_one_hot = to_categorical(y_test, num_classes=10)
  score = model.evaluate(x=x_test, y=y_test_one_hot, verbose=0)
  return score

def process_boston():
  import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  scale = StandardScaler()
  dset = pd.read_csv("/kaggle/input/boston-housing-dataset/HousingData.csv")
  dset = dset.astype({
    'CHAS':float,
    'RAD':float,
    'TAX':float
  })
  mean_value1 = dset['CRIM'].mean()
  mean_value2 = dset['ZN'].mean()
  mean_value3 = dset['INDUS'].mean()
  mean_value4 = dset['CHAS'].mean()
  mean_value5 = dset['AGE'].mean()
  mean_value6 = dset['LSTAT'].mean()
  dset['CRIM'].fillna(value=mean_value1, inplace=True)
  dset['ZN'].fillna(value=mean_value2, inplace=True)
  dset['INDUS'].fillna(value=mean_value3, inplace=True)
  dset['CHAS'].fillna(value=mean_value4, inplace=True)
  dset['AGE'].fillna(value=mean_value5, inplace=True)
  dset['LSTAT'].fillna(value=mean_value6, inplace=True)

  x = np.array(dset.drop('MEDV',axis=1))
  y= np.array(dset[['MEDV']])
  X = scale.fit_transform(x)
  Y = scale.fit_transform(y)
  xtrain,xtest,ytrain,ytest= train_test_split(X,Y,test_size=0.20)
  return xtrain,xtest,ytrain,ytest

def create_boston_model():
  from keras.models import Sequential
  from keras.layers import Dense
  model = Sequential()
  model.add(Dense(units=12, input_dim=X.shape[1], activation='relu'))
  model.add(Dense(units=6, activation='relu'))
  model.add(Dense(units=1, activation='linear'))
  model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['mse','mae'])
  return model

def train_boston(model, xtrain, ytrain):
  from sklearn.model_selection import train_test_split
  import tensorflow as tf
  from tensorflow import keras
  from keras.layers import Dropout
  from keras import metrics
  from keras import layers
  from keras.metrics import MeanSquaredError ,MeanAbsoluteError
  history = model.fit(xtrain,ytrain,batch_size=10,epochs=100,validation_split=0.3)
  return history, model

def evaluate_boston(model):
  from sklearn.metrics import r2_score
  acc = model.predict(xtest)
  return (r2_score(ytest, acc))*100

def train(dataset_path):
  dataset = load_dataset_from_file(dataset_path)
  nameOfDataset = dataset.split("/")[-1]
  
  if nameOfDataset == "mnist":
    s = time.time()
    xtrain,xtest,ytrain,ytest = process_mnist()
    run_time_process = int(time.time() - s)
  
    s = time.time()
    model = create_mnist_model()
    train_history, model = train_mnist(model, xtrain,ytrain)
    run_time_train = int(time.time() - s)
    
    s = time.time()
    test_history = evaluate_mnist(model, xtest, ytest)
    run_time_process = int(time.time() - s)
  
  elif nameOfDataset == "boston":
    s = time.time()
    xtrain,xtest,ytrain,ytest = process_boston()
    run_time_process = int(time.time() - s)
  
    s = time.time()
    model = create_boston_model()
    train_history, model = train_boston(model,xtrain,ytrain)
    run_time_train = int(time.time() - s)
  
    s = time.time()
    test_history = evaluate_boston()
    run_time_process = int(time.time() - s)

def predict(testData):
  
  testData = load_dataset_from_file(testData)
  nameOfData = testData.split("/")[-1]

  if nameOfData == 'mnist':
    #to complete

  elif nameOfData=='boston':
    #to complete
      
    

