{-|
Module      : TypedFlow.Layers.Core
Description : Core layers and combinators.
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeInType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE PatternSynonyms #-}

module TypedFlow.Layers.Core
  (
    -- * Dense
    DenseP(..), dense, (#),
    -- * Dropout
    DropProb(..), mkDropout, mkDropouts,
    -- * Embedding
    EmbeddingP(..), embedding, 
    -- * Convolutional
    ConvP(..), conv, {-convValid,-} maxPool1D, maxPool2D)

where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import qualified Prelude
import GHC.TypeLits
-- import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Python
import Control.Monad.State (gets)
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)
import Data.Monoid ((<>))
---------------------
-- Linear functions


-- type (a ⊸ b) = DenseP Float32 a b

-- | A dense layer is a linear function form a to b: a transformation matrix and a bias.
data DenseP t a b = DenseP {denseWeights :: Tensor '[a,b] (Flt t)
                           ,denseBiases  :: Tensor '[b] (Flt t)}

-----------------------
-- Feed-forward layers

-- | Parameters for the embedding layers
newtype EmbeddingP numObjects embeddingSize t = EmbeddingP (Tensor '[numObjects, embeddingSize] ('Typ 'Float t))

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => KnownTensors (EmbeddingP numObjects embeddingSize b) where
  travTensor f s (EmbeddingP p) = EmbeddingP <$> travTensor f s p

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbeddingP numObjects embeddingSize b) where
  defaultInitializer = EmbeddingP (randomUniform (-0.05) 0.05)

-- | embedding layer
embedding :: ∀ embeddingSize numObjects t. KnownNat embeddingSize => KnownNat numObjects =>
             EmbeddingP numObjects embeddingSize t -> Tensor '[] Int32 -> Tensor '[embeddingSize] ('Typ 'Float t)
embedding (EmbeddingP param) input = gather param input



instance (KnownNat a, KnownNat b, KnownBits t) => KnownTensors (DenseP t a b) where
  travTensor f s (DenseP x y) = DenseP <$> travTensor f (s<>"_w") x <*> travTensor f (s<>"_bias") y

instance (KnownNat n, KnownNat m, KnownBits b) => ParamWithDefault (DenseP b n m) where
  defaultInitializer = DenseP glorotUniform (truncatedNormal 0.1)

-- | Dense layer (Apply a linear function)
(#), dense :: ∀m n t. KnownNat n => KnownNat m => KnownBits t => DenseP t n m -> Tensor '[n] (Flt t) -> Tensor '[m] (Flt t)
(DenseP weightMatrix bias) # v = (weightMatrix ∙ v) + bias

dense = (#)

-- | A drop probability. (This type is used to make sure one does not
-- confuse keep probability and drop probability)
data DropProb = DropProb Float

-- | Generate a dropout function. The mask applied by the returned
-- function will be constant for any given call to mkDropout. This
-- behavior allows to use the same mask in the several steps of an
-- RNN.
mkDropout :: forall s t. KnownShape s => KnownBits t => DropProb -> Gen (Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t))
mkDropout (DropProb dropProb) = do
  let keepProb = 1.0 Prelude.- dropProb
  isTraining <- gets genTrainingPlaceholder
  mask <- assign (if_ isTraining
                   (floor (randomUniform keepProb (1 Prelude.+ keepProb)) ⊘ constant keepProb)
                   ones)
  return (mask ⊙)

newtype EndoTensor t s = EndoTensor (Tensor s t -> Tensor s t)

-- | Generate a dropout function for an heterogeneous tensor vector.
mkDropouts :: KnownBits t => KnownLen shapes => All KnownShape shapes => DropProb -> Gen (HTV ('Typ 'Float t) shapes -> HTV ('Typ 'Float t) shapes)
mkDropouts d = appEndoTensor <$> mkDropouts' typeSList where
   mkDropouts' :: forall shapes t. KnownBits t => All KnownShape shapes =>
                  SList shapes -> Gen (NP (EndoTensor ('Typ 'Float t)) shapes)
   mkDropouts' LZ = return Unit
   mkDropouts' (LS _ rest) = do
     x <- mkDropout d
     xs <- mkDropouts' rest
     return (EndoTensor x :* xs)

   appEndoTensor :: NP (EndoTensor t) s -> HTV t s -> HTV t s
   appEndoTensor Unit Unit = Unit
   appEndoTensor (EndoTensor f :* fs) (F x :* xs) = F (f x) :* appEndoTensor fs xs


------------------------
-- Convolutional layers

data ConvP t outChannels inChannels filterSpatialShape
  = ConvP (T (filterSpatialShape ++ '[inChannels,outChannels])  ('Typ 'Float t)) (T '[outChannels] ('Typ 'Float t))

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  ParamWithDefault (ConvP t outChannels inChannels filterSpatialShape) where
  defaultInitializer = prodHomo @filterSpatialShape @'[inChannels, outChannels] $
                       prodAssoc @(Product filterSpatialShape) @inChannels @outChannels $
                       knownAppend @filterSpatialShape @'[inChannels,outChannels] $
                       knownProduct @filterSpatialShape $
                       ConvP (reshape i) (constant 0.1)
    where i :: T '[Product filterSpatialShape*inChannels,outChannels] (Flt t)
          i = knownProduct @filterSpatialShape glorotUniform

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  KnownTensors (ConvP t outChannels inChannels filterSpatialShape) where
  travTensor f s (ConvP x y) = knownAppend @filterSpatialShape @'[inChannels,outChannels] $
          ConvP <$> travTensor f (s<>"_filters") x <*> travTensor f (s <> "_biases") y

-- | Size-preserving convolution layer
conv :: forall outChannels filterSpatialShape inChannels t.
               KnownNat inChannels => KnownNat outChannels => KnownShape filterSpatialShape => KnownBits t
            => Length filterSpatialShape <= 3
            => ConvP t outChannels inChannels filterSpatialShape
            -> T (filterSpatialShape ++ '[inChannels]) ('Typ 'Float t)
            -> T (filterSpatialShape ++ '[outChannels]) ('Typ 'Float t)
conv (ConvP filters bias) input = mapT (+bias) (convolution @outChannels @filterSpatialShape @inChannels input filters)

-- -- | Convolution layers with no padding (applying the filter only on
-- -- positions where the input is fully defined, aka "VALID" in
-- -- tensorflow.)
-- convValid :: forall outChannels filterSpatialShape inChannels s t.
--                   ((1 + Length filterSpatialShape) ~ Length s,
--                    Length filterSpatialShape <= 3,
--                    KnownLen filterSpatialShape) -- the last dim of s is the batch size
--           => ConvP t outChannels inChannels filterSpatialShape -- ^ Parameters
--           -> T ('[inChannels] ++ AddSpatialDims s filterSpatialShape) ('Typ 'Float t) -- ^ input
--           -> (T ('[outChannels] ++ s) ('Typ 'Float t))
-- convValid (ConvP filters bias) input = convolutionValid input filters + bias


-- | x by y maxpool layer.
maxPool2D :: forall windowx windowy batch height width channels t.
             (KnownNat windowx, KnownNat windowy) =>
             T '[channels,width*windowx,height*windowx,batch] (Flt t) -> T '[channels,width,height,batch] (Flt t)
maxPool2D (T value) = T (funcall "tf.nn.max_pool" [value
                                                  ,showShape @'[1,windowx,windowy,1]
                                                  ,showShape @'[1,windowx,windowy,1]
                                                  ,named "padding" (str "SAME") ])

-- | maxpool layer. window size is the first type argument.
maxPool1D :: forall window batch width channels t.
             (KnownNat window) =>
             T '[channels,width*window,batch] (Flt t) -> T '[channels,width,batch] (Flt t)
maxPool1D (T value) = T (funcall "tf.nn.pool" [named "input" value
                                              ,named "window_shape" (showShape @'[1,window,1])
                                              ,named "pooling_type" (str "MAX")
                                              ,named "padding" (str "SAME")
                                              ])

