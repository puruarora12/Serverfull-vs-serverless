import time

def process_mnist():
  pass

def train_mnist():
  pass

def evaluate_mnist():
  pass

def process_boston():
  pass

def train_boston():
  pass

def evaluate_boston():
  pass


if data == "mnist":
  s = time.time()
  process_mnist()
  run_time_process = int(time.time() - s)

  s = time.time()
  train_mnist()
  run_time_train = int(time.time() - s)
  
  s = time.time()
  evaluate_mnist()
  run_time_process = int(time.time() - s)

elif data == "boston":
  s = time.time()
  process_boston()
  run_time_process = int(time.time() - s)

  s = time.time()
  train_boston()
  run_time_train = int(time.time() - s)

  s = time.time()
  evaluate_boston()
  run_time_process = int(time.time() - s)

