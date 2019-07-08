{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TypeApplications #-}
-- TODO hoist to package.yaml

module Integration.PythonSpec where

import TypedFlow
import TypedFlow.Python
import Integration.Helpers

import Test.Tasty
import Test.Tasty.Hspec
import Test.Tasty.SmallCheck

(&) = flip ($) -- TODO

spec_characterization :: Spec
spec_characterization = parallel $ do
  -- TODO break apart, simplify
  it "compiles and trains a simple classifier: 'is the input sorted?'" $ do
    -- Input: T '[5] Float32
    -- Output: T '[2] Float32 (sorted | not sorted)
    let model :: Gen (Model AccuracyShape '[5] Float32 '[2] '[2] '[] Float32)
        model = do
          w1 <- parameterDefault "w1"
          w2 <- parameterDefault "w2"
          w3 <- parameterDefault "w3"

          return $ \input gold ->
            input & dense @10 w1 & batchNorm & relu
                  & dense @10 w2 & batchNorm & relu
                  & dense @2 w3
                  & (`categoricalDistribution` gold)
        pythonFile = "spec/generated/sorted_classifier_tyf.py"
    generateFile pythonFile $ compile @AccuracyShape @100 defaultOptions model
    
    -- Training
    output <- runPython "sorted_classifier" [py|
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
    |]
    pure ()
