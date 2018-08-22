{-|
Module      : TypedFlow.Haskell
Description : Generation of computation graph using tensorflow haskell. 
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental

-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Haskell where

import Data.Type.Equality
import Data.List (genericReplicate)
import GHC.TypeLits
import Control.Monad.State
import TypedFlow.Types
import TypedFlow.Types.Proofs
import TypedFlow.Abstract (newId, permToFun)
import TypedFlow.Memo
import System.Mem.StableName
import System.IO.Unsafe

import qualified Data.Int as Backend

import qualified TensorFlow.Core        as Backend
import qualified TensorFlow.GenOps.Core
import qualified TensorFlow.Minimize    as Backend
import qualified TensorFlow.Ops         as Backend
-- import qualified TensorFlow.Variable    as Backend

import qualified Data.IntMap as IM
import Data.IntMap (IntMap)

type BackendShape = BackendTensor ('Typ 'Int 'B32)
type BackendTensor t = Backend.Tensor Backend.Build (HaskType t)
type BackendVariable t = Backend.Tensor Backend.Ref (HaskType t)

shapeFromType :: ∀ (s :: Shape). KnownShape s => BackendShape
shapeFromType = shapeVector (typeSShape @s)

-- | Show a shape, but "None" is replaced by "-1"
shapeVector :: forall (s::Shape) proxy. All KnownNat s => SList' proxy s -> BackendShape
shapeVector s = shapeFromList (shapeToList'' s)

permToTensor :: SShape s -> Permutation s t -> Backend.Tensor Backend.Build Backend.Int32
permToTensor s p = Backend.vector (map (fromInteger . permToFun p) [0.. sListLength s])

shapeFromList :: [Integer] -> BackendShape
shapeFromList = Backend.vector . map convertNone

showShapeLen :: ∀ (s::Shape). KnownLen s => Backend.Int32
showShapeLen = fromIntegral (listTypeLen @ s)

convertNone :: Num a => Integer -> a
convertNone n = (if n == 514229 then (-1) else fromIntegral n)

-- runWithFeeds

data BT (s :: Shape) (t :: Typ) where
  BT :: forall s t v. (Backend.Tensor v (HaskType t)) -> BT s t

data HState = HState {genVars :: IntMap Var
                     ,genPureTable :: SNMap22 Shape Typ T BT
                     }

type BM a = Backend.BuildT (StateT HState (State GState)) a

data Var = forall s t v. Var (SShape s) (STyp t) (Backend.Tensor v (HaskType t))

initializedVariable :: forall s a. KnownShape s => KnownTyp a => T s a -> BM (Ref s a)
initializedVariable initVal = do
  BT i <- interpretPure initVal
  x <- lift (lift newId)
  v <- backendTensor (typeSTyp @a) $ Backend.initializedVariable i
  let var = (Var (typeSShape @s) (typeSTyp @a) v)
  lift (modify $ \HState{..} -> HState {genVars = IM.insert (fromIntegral x) var genVars,..})
  return (Ref (fromIntegral x) typeSShape typeSTyp )

placeholder :: forall s a. SShape s -> STyp a -> BM (Ref s a)
placeholder s t = do
  x <- lift (lift newId)
  ph <- backendTensor t $ Backend.placeholder (Backend.Shape (map convertNone $ shapeToList' s))
  let var = (Var s t ph)
  lift (modify $ \HState{..} -> HState {genVars = IM.insert (fromIntegral x) var genVars,..})
  return (Ref (fromIntegral x) s t )

interpGen :: Gen a -> BM a
interpGen (GPReturn x) = return x
interpGen (GPVariable _trainable _name initVal) = initializedVariable initVal
interpGen (GPPlaceholder s t _name) = placeholder s t
interpGen (GPModify _ _) = error "GPModify: TODO"
interpGen (GPState f) = lift (lift (state f))
interpGen (GPBind a b) = do x <- interpGen a
                            interpGen (b x)

listProxyLen :: forall proxy s. KnownLen s => proxy s -> Integer
listProxyLen _ = listTypeLen @s

-- genDistr :: forall s s0 t. KnownTyp t => Distribution s t -> SShape s0 -> SShape s -> DOC
-- genDistr d sh s1 = case d of
--   TruncatedNormalD stddev -> funcall "tf.truncated_normal"
--     [showSShape (sh .+. s1), named "stddev" (float stddev), named "dtype" (showTyp @t)]
--   UniformD low high -> funcall "tf.random_uniform" [showSShape (sh .+. s1)
--                                 ,named "minval" (float low)
--                                 ,named "maxval" (float high)
--                                 ,named "dtype" (showTyp @t)]
--   OrthogonalD ->
--     funcall' (funcall "tf.orthogonal_initializer" [named "dtype" (showTyp @t)]) [named "shape" (showSShape (sh .+. s1))]


knownNumeric :: forall t k. KnownNumeric t => (KnownTyp t => Num (HaskType t) => Backend.OneOf '[Backend.Int32, Float, Double] (HaskType t) => k) -> k
knownNumeric k = case kindVal @(TypKind t) of
  SFloat -> case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> k
  SBool -> error "TFNumeric bug"
  SInt -> case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> error "missing in tensorflow: int64 is not supported in matmul T_T"

backendTensor :: STyp t ->  (Backend.TensorType (HaskType t) => k) -> k
backendTensor (STyp SFloat SB32 Refl) k = k
backendTensor (STyp SInt SB64 Refl) k = k
backendTensor (STyp SBool _ Refl) k = k
backendTensor (STyp SFloat SB64 Refl) k = k
backendTensor (STyp SInt SB32 Refl) k = k

backendTensor' :: forall t k proxy. KnownTyp t => proxy t -> (Backend.TensorType (HaskType t) => k) -> k
backendTensor' _ = backendTensor (typeSTyp @t)

runUnOp :: UnOp s1 t s2 u -> BackendTensor t -> BackendTensor u
runUnOp _ = error "todo"

interpretPure :: forall s t. KnownTyp t => KnownShape s => T s t -> BM (BT s t)
interpretPure x = do
  let sn = unsafePerformIO $ makeStableName x
  mv <- snMap22Lookup sn <$> lift (gets genPureTable)
  case mv of
    Just v -> return v
    Nothing -> do
      e  <- interpretPure' (\s x' -> knownSShape s $ interpretPure x') typeSShape x
      lift $ modify (\g -> g {genPureTable = (snMap22Insert (KV sn e)) (genPureTable g)})
      return e

interpNilOp :: Backend.TensorType (HaskType t) => NilOp s t -> BM (BT s t)
interpNilOp = \case
  Constant c -> return $ BT $ Backend.scalar c
  Range _ -> _
  Variable (Ref r sr tr) -> do
     tbl <- lift (gets genVars)
     case IM.lookup r tbl of
       Just (Var sx tx x) -> case (testEq sx sr, testEq tx tr) of
          (Just Refl, Just Refl) -> return (BT x)
          _ -> error "panic: variable does not have the expected type"
       _ -> error "panic: variable not found" 

interpretPure' :: forall s t. KnownTyp t => (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> BM (BT s' t')) -> SShape s -> T s t -> BM (BT s t)
interpretPure' rec sR = knownSShape sR $ backendTensor (typeSTyp @t) $ \case
  Unbroadcast{} -> error "broadcasting operation did not complete!"
  DirectBroadcast s0 s1 s2 s3 x -> do
    BT recx <- rec (s0 .+. s2) x
    return $ BT $ TensorFlow.GenOps.Core.broadcastTo recx (shapeFromList
                                        (concat [shapeToList' s0, genericReplicate (sListLength s1) 1
                                                ,shapeToList' s2, genericReplicate (sListLength s3) 1 ]))
   --  Noise noiseId s0 s1 x -> do
   --    return $ (genDistr x s0 s1) <+> (text "# " <> integer noiseId)
  T op -> interpNilOp op
  -- T x _ -> x $ Backend.Shape $ map fromIntegral $ shapeToList' sR
   --  If c x y -> do
   --    rc <- rec typeSShape c
   --    rx <- rec typeSShape x
   --    ry <- rec typeSShape y
   --    return (func "tf.cond" [rc] [("true_fn", lambda0 rx) ,("false_fn", lambda0 ry) ,("strict","True")])
   --    where lambda0 z = text "lambda: " <> z
  -- Where c x y -> let rc = rec typeSShape c
  --                    rx = rec typeSShape x
  --                    ry = rec typeSShape y
  --                in Backend.select rc rx ry
  -- UnOp operation s0 s1 _s2 x ->
  --   let recx = rec (s0 .+. s1) x
  --   in runUnOp operation recx
 --   return $ case operation of
 --    Axis1Op op args n -> func op [recx] ((axisName,integer (sListLength s0 + n)):args)
 --      where axisName = if op == "tf.nn.softmax" then "dim" else "axis" -- use dim before TF 1.5
 --    Simple1Op op args -> funcall op (recx:args)
 --    SliceOp lo hi -> recx <> list (replicate (fromIntegral (sListLength s0)) (text ":") ++ [integer lo <> text ".." <> integer hi])
 --    IndexOp axis ix -> recx <> list (replicate (fromIntegral (axis + sListLength s0)) (text ":") ++ [integer ix])
  -- MatMul s0 a b c x y  -> let recx = rec (s0 .+. (:*) a ((:*) b Unit)) x
  --                             recy = rec (s0 .+. (:*) b ((:*) c Unit)) y
  --                         in knownNumeric @t $ Backend.matMul recx recy
 --  BinOp operation s0 s1 s2 _s3 x y -> do
 --   recx <- rec (s0 .+. s1) x
 --   recy <- rec (s0 .+. s2) y
 --   return $ case operation of
 --     Axis2Op op n -> funcall op  [list [recx,recy], named "axis" (integer (sListLength s0 + n))]
 --     Simple2Op op Nothing -> funcall op [recx, recy]
 --     Simple2Op op (Just (nx,ny)) -> func op [] [(nx,recx), (ny,recy)]
 --  ReshapeFrom s t -> do
 --    rt <- rec s t
 --    return (funcall "tf.reshape" [rt, showShapeMinus sR])
 --  Stack s0 _m s1 (V xs) -> do
 --    rxs <- mapM (rec (s0 .+. s1)) xs
 --    return (funcall "tf.stack" [list rxs, text "axis=" <> integer (sListLength s0)])
  -- Transpose s p x -> Backend.transpose (rec s x) (permToTensor s p)
 --  Gather indexShape s0 m s1 x ix -> do
 --    rx <- rec (s0 .+. ((:*) m s1)) x
 --    rix <- rec indexShape ix
 --    return (func "tf.gather" [rx, rix] [])
 --  GatherND containerShape elementShape indexShape x ix -> do
 --    rx <- rec (containerShape .+. elementShape) x
 --    rix <- rec (indexShape *: (sListLenAsNat containerShape)) ix
 --    return (func "tf.gather_nd" [rx, rix] [])
 --  Convolution bs inChans outChans filterShape s0 x filters -> do
 --    recx <- rec ((:*) bs (s0 *: inChans)) x
 --    recFilters <- rec (filterShape .+. ((:*) inChans ((:*) outChans Unit))) filters
 --    return (func "tf.nn.convolution" [recx, recFilters] [("padding",text (show ("SAME"::String))),("data_format", text (show dataFormat))])
 --   where dataFormat = case sListLength filterShape of
 --           1 -> ("NWC" :: String)
 --           2 -> "NHWC"
 --           3 -> "NDHWC"
 --           _ -> error "convolution: more than 3 spatial dimensions are not supported!"
 --  Pool bs window typ numChans outSpatial x -> do
 --     rx <- rec ((:*) bs (zipWithMulSShapes window outSpatial .+. (:*) numChans Unit)) x
 --     return (func "tf.nn.pool"
 --                  [rx, showSShape window, typ', text (show ("SAME" :: String))]
 --                  [("strides", showSShape window)])
 --   where typ' = text $ (show $ case typ of MaxPool -> "MAX"; AvgPool -> "AVG" :: String)
 -- -- where rec :: forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> DOC
 -- --       rec = generatePure' 
