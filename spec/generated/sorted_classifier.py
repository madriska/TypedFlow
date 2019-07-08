import numpy as np
import tensorflow as tf
import typedflow_rts as tyf
import random
from .sorted_classifier_tyf import mkModel

def dataGenerator(bs):
  for _ in range(1000):
    xs = []
    ys = []
    for _ in range(bs):
      if random.random() < 0.5:
        # positive examples
        v = [random.uniform(-10000, 10000) for _ in range(5)]
        xs.append(sorted(v))
        ys.append([1.0, 0.0])
      else:
        # negative examples
        v = None
        while True:
          v = [random.uniform(0, 100) for _ in range(5)]
          if list(v) != list(sorted(v)):
            break
        xs.append(v)
        ys.append([0.0, 1.0])
    yield {"x": xs, "y": ys}

sess = tf.Session()
model = mkModel(optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
tyf.initialize_params(sess, model)
# TODO the shape should be injected automatically here
stats = tyf.train(sess, model, np.array([0.,0.]), dataGenerator)

correct, total = stats[-1]['train']['metrics']
acc = correct / total

print(f"Accuracy: {acc * 100}%")
if acc < 0.999:
  raise Exception(f"Expected 99.9% or greater accuracy but got: {acc * 100}%")
