{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module MNIST where

import TypedFlow
import TypedFlow.Python

(&) = flip ($)

is :: forall s t. T s t -> T s t
is = id

mnist :: Gen (Model '[784] Float32 '[10] '[10] '[] Float32)
mnist = do
  filters1 <- parameterDefault "f1"
  filters2 <- parameterDefault "f2"
  w1 <- parameterDefault "w1"
  w2 <- parameterDefault "w2"
  
  return $ \input gold ->
    input                              & reshape @'[28,28,1]
    & conv @32 @'[5,5] filters1 & relu & is      @'[28,28,32]
    & maxPool2D @2 @2                  & is      @'[14,14,32] 
    & conv @64 @'[5,5] filters2 & relu & is      @'[14,14,64]
    & maxPool2D @2 @2                  & is      @'[7,7,64]
                                       & reshape @'[3136]
    & dense @1024 w1 & relu
    & dense @10 w2
    & (`categoricalDistribution` gold)

main :: IO ()
main = do
  generateFile "mnist_model.py" (compile @100 defaultOptions mnist)
  putStrLn "done!"

-- >>> main
-- Parameters (total 3274634):
-- w2_bias: T [10] tf.float32
-- w2_w: T [1024,10] tf.float32
-- w1_bias: T [1024] tf.float32
-- w1_w: T [3136,1024] tf.float32
-- f2_biases: T [64] tf.float32
-- f2_filters: T [5,5,32,64] tf.float32
-- f1_biases: T [32] tf.float32
-- f1_filters: T [5,5,1,32] tf.float32
-- done!


-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:

