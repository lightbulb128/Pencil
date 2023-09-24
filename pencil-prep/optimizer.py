import math
import numpy as np
import pickle

DANGEROUS_GRAD_THRESHOLD = 4
DANGEROUS_VALUE_THRESHOLD = 0.5

def wipe_dangerous_grad(parameters, threshold=DANGEROUS_GRAD_THRESHOLD):
  for parameter in parameters:
    parameter.grad = np.where(np.abs(parameter.grad) > threshold, 0, parameter.grad)

def wipe_dangerous_value(parameters, threshold=DANGEROUS_VALUE_THRESHOLD):
  for parameter in parameters:
    parameter.value = np.where(parameter.value > threshold, threshold, parameter.value)
    parameter.value = np.where(parameter.value < -threshold, -threshold, parameter.value)

class Optimizer:
  def zero_grad(self): 
    for each in self.parameters:
      each.zero_grad()
  def step(self): pass
  def save(self, f): pass
  def load(self, f): pass
  def reset_parameters(self, parameters): 
    self.parameters = parameters
  def report_max_values(self):
      max_value = 0
      max_i = -1
      for i, parameter in enumerate(self.parameters):
        max_this = np.max(np.abs(parameter.grad))
        if max_this > max_value: 
          max_value = max_this
          max_i = i
      print("Optimizer: maxgrad =", max_value, f"({max_i})")
      max_value = 0
      max_i = -1
      for i, parameter in enumerate(self.parameters):
        max_this = np.max(np.abs(parameter.value))
        if max_this > max_value: 
          max_value = max_this
          max_i = i
      print("Optimizer: maxvalue =", max_value, f"({max_i})")

class SGD(Optimizer):
  def __init__(self, parameters, learning_rate):
    self.parameters = parameters
    self.learning_rate = learning_rate
    self.weight_decay = 0
    self.silent = False

  def silence(self):
    self.silent = True
    
  def step(self):
    
    if not self.silent:
      self.report_max_values()

    for each in self.parameters: 
      each.value *= (1 - self.learning_rate * self.weight_decay)
      each.value -= each.grad * self.learning_rate

class SGDMomentum(Optimizer):
  def __init__(self, parameters, learning_rate, momentum, dampening=0):
    self.parameters = parameters
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.dampening = dampening
    self.start_up = True
    self.weight_decay = 0
    self.silent = False

  def silence(self):
    self.silent = True

  def step(self):
    
    if not self.silent:
      self.report_max_values()

    # keep it safe
    wipe_dangerous_grad(self.parameters)

    if self.start_up:
      self.b = [parameter.grad.copy() for parameter in self.parameters]
      self.start_up = False
    else:
      new_b = []
      for b, parameter in zip(self.b, self.parameters):
        new_b.append(b * self.momentum + (1-self.dampening) * parameter.grad)
      self.b = new_b
    max_value = 0
    max_i = -1
    for i, (b, parameter) in enumerate(zip(self.b, self.parameters)):
      parameter.value -= self.learning_rate * b
    
    wipe_dangerous_value(self.parameters)

  def save(self, f):
    pickle.dump(self.start_up, f)
    pickle.dump(self.b, f)
  def load(self, f):
    self.start_up = pickle.load(f)
    self.b = pickle.load(f)

class Adam(Optimizer):
  def __init__(self, parameters, learning_rate, betas=(0.9, 0.999), eps=1e-8):
    self.parameters = parameters
    self.learning_rate = learning_rate
    self.beta1 = betas[0]
    self.beta2 = betas[1]
    self.eps = eps
    self.time = 0
    self.m = [np.zeros_like(a.value, dtype=a.value.dtype) for a in parameters]
    self.v = [np.zeros_like(a.value, dtype=a.value.dtype) for a in parameters]
    self.weight_decay = 0

  def silence(self):
    pass

  def step(self):

    if not self.silent:
      self.report_max_values()

    self.time += 1
    for i, parameter in enumerate(self.parameters):
      self.m[i] *= self.beta1
      self.m[i] += (1-self.beta1) * parameter.grad
      self.v[i] *= self.beta2
      self.v[i] += (1-self.beta2) * parameter.grad**2
      bcor1 = 1 - self.beta1 ** self.time
      bcor2 = 1 - self.beta2 ** self.time

      step_size = self.learning_rate / bcor1
      bcor2sqrt = math.sqrt(bcor2)
      denom = np.sqrt(self.v[i]) / bcor2sqrt + self.eps

      # print("optim m", self.m[i].flatten()[0])
      # print("optim d", denom.flatten()[0])

      parameter.value -= step_size * self.m[i] / denom
    wipe_dangerous_grad(self.parameters, 4)


  def save(self, f):
    pickle.dump(self.time, f)
    pickle.dump(self.m, f)
    pickle.dump(self.v, f)
  def load(self, f):
    self.time = pickle.load(f)
    print(self.time)
    self.m = pickle.load(f)
    self.v = pickle.load(f)
        