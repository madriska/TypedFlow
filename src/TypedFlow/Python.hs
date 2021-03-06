{-|
Module      : TypedFlow.Python
Description : Python-generation Functions 
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
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Python (compile, compileFn, compileGen, generateFile, rtsCode) where

import Data.ByteString (ByteString)
import Data.Char (toLower)
import Data.FileEmbed (embedFile)
import Data.Proxy
import Data.List (genericReplicate, isPrefixOf)
import GHC.TypeLits
import Control.Monad.State
import TypedFlow.Types
import TypedFlow.Abstract (newId, permToFun,unopInputShape)
import TypedFlow.Types.Proofs
import TypedFlow.Memo
import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum,Options)
import qualified Text.PrettyPrint.Compact as PP
import qualified Data.Map as M
import TypedFlow.Learn

first :: (t -> a) -> (t, b) -> (a, b)
first f (x,y) = (f x,y)

paramShape' :: VarInfo -> [Integer]
paramShape' (VarInfo _ s _ _) = shapeToList' s

paramDType ::  VarInfo -> Typ
paramDType (VarInfo _ _ t _) = sTypTyp t

paramName :: VarInfo -> String
paramName (VarInfo nm _ _ _) = nm

generateFile :: String -> Python () -> IO ()
generateFile fname g = do
  putStrLn ("Parameters (total " ++ show (sum [product (paramShape' p) | p <- params]) ++ "):")
  forM_ params printParam
  writeFile fname output
  where (output,params) = generate g
        printParam p = putStrLn (paramName p ++ ": " ++ "T " ++ render (showShape' (paramShape' p))  ++ " " ++ showT (paramDType p))

named :: String -> DOC -> DOC
named fname x = text (fname <> "=") <> x

genFun :: forall b. String -> [DOC] -> Python b -> Python b
genFun name args body = do
  gen (text "def " <> text name <> tuple args <> text ":")
  withDOC (\b -> text "  " <> b) body


showTyp :: forall t. KnownTyp t => DOC
showTyp = text (showT (typVal @t))

showSTyp :: forall t. STyp t -> DOC
showSTyp t = knownTyp t $ showTyp @t

showT :: Typ -> [Char]
showT (Typ Bool _) = "tf.bool"
showT (Typ k l) = "tf." ++ map toLower (show k) ++ drop 1 (show l)

showShape' ::  [Integer] -> DOC
showShape' s = list (map (showDim' "None") s)

showShape :: ∀ (s :: Shape). All KnownNat s => SList s -> DOC
showShape s = showShape' (shapeToList'' s)

showSShape :: ∀ (s :: Shape). SShape s -> DOC
showSShape s = showShape' (shapeToList' s)

showShapeType :: ∀ (s :: Shape). KnownShape s => DOC
showShapeType = showSShape (typeSShape @s)

showShapeTypeAssert :: ∀ (s :: Shape) t. KnownShape s => Ref s t -> DOC
showShapeTypeAssert ref =
  "assert " <> pyVarRepr ref <> ".shape.as_list() == " <> showShapeType @s

-- | Show a shape, but "None" is replaced by "-1"
showShapeMinus :: forall (s::Shape) proxy. All KnownNat s => SList' proxy s -> DOC
showShapeMinus s = list (map (showDim' "-1") (shapeToList'' s))

showShapeLen :: ∀ (s::Shape). KnownLen s => DOC
showShapeLen = (text . show) (listTypeLen @ s)

showDim' :: String -> Integer -> DOC
showDim' none n = text (if n == 514229 then none else show n)

showDimM :: forall n. KnownNat n => DOC
showDimM = showDim' "-1" (natVal (Proxy @ n))

showDim :: forall n. KnownNat n => DOC
showDim = showDim' "None" (natVal (Proxy @ n))

showDimS :: forall n. Sat KnownNat n -> DOC
showDimS Sat = showDim @n

gen :: DOC -> Python ()
gen s = modify $ \PyState{..} -> PyState {genText=genText $$ s,..}

setGen :: DOC -> Python ()
setGen d = modify $ \PyState{..} -> PyState {genText=d,..}

-- | Output an assignment statement assigning "y" to "x".
(<--) :: Ref s t -> UntypedExpression -> Python ()
x <-- y = gen (pyVarRepr x <> text "=" <>  y)

-- | save an intermediate result to a variable and save it to
-- genAssignTable for future re-use.
cache :: forall s t. KnownTyp t => KnownShape s => DOC  -> Python DOC
cache x = do
  let x' = renderWith (PP.Options 92 (const id)) x
  mcache <- M.lookup x' <$> gets genAssignTable
  case mcache of
    Just y -> return y
    Nothing -> do
      v <- newPyVar @s @t

      v <-- x
      gen (showShapeTypeAssert @s v)
      modify $ (\g -> g {genAssignTable = M.insert x' (pyVarRepr v) (genAssignTable g)})
      return (pyVarRepr v)

newPyVar' :: forall s t. SShape s -> STyp t -> Python (Ref s t)
newPyVar' s t = knownSShape s ?> (knownTyp t $ newPyVar @s @t)

newPyVar :: forall s t. KnownShape s => KnownTyp t => Python (Ref s t)
newPyVar = do
  n <- lift newId
  return $ Ref (fromIntegral n) typeSShape typeSTyp

pyVarRepr :: Ref s t -> DOC
pyVarRepr (Ref n _ _) = text ("var" <> show n)

pyLitRepr :: forall s t. KnownTyp t => LiteralTensor s t -> DOC
pyLitRepr (LitA1 xs) = funcall "tf.convert_to_tensor" [pyArrayOf (pretty @t) xs]
pyLitRepr (LitA2 xss) =
  funcall "tf.reshape"
    [ funcall "tf.convert_to_tensor" [pyArrayOf (pyArrayOf (pretty @t)) xss]
    , showShapeMinus (typeSShape @s)
    ]

pyArrayOf :: (a -> DOC) -> [a] -> DOC
pyArrayOf f = brackets . sep . punctuate comma . map f

tuple :: [DOC] -> DOC
tuple = parens . sep . punctuate comma

dict :: [(String,DOC)] -> DOC
dict xs = encloseSep "{" "}" "," [text (show k) <> ":" <> v | (k,v) <- xs]

funcall :: String -> [DOC] -> DOC
funcall = funcall' . text

funcall' :: DOC -> [DOC] -> DOC
funcall' f args = hangWith "" 2 (f <> "(") (as <> ")")
  where as = sep (punctuate comma args)

func :: String -> [DOC] -> [(String,DOC)] -> DOC
func fname positional namedArgs = funcall fname (positional ++ map (uncurry named) namedArgs )

withDOC :: forall a. (DOC -> DOC) -> Python a -> Python a
withDOC f g = do
  before <- gets genText
  setGen mempty
  x <- g
  after <- gets genText
  setGen (before $$ f after)
  return x

assignAny :: UntypedExpression -> Python UntypedExpression
assignAny x = do
  v <- newPyVar @'[] @Float32
  v <-- x
  return (pyVarRepr v)

generate :: Python () -> (String,[VarInfo])
generate s = (renderWith (PP.Options 92 (const id)) genText, genParams)
  where (PyState{..},GState{..}) = runState (execStateT s initPyState) initialGstate
        initPyState = PyState {genPureTable = mempty
                              ,genAssignTable = mempty
                              ,genText = mempty}

generatePure :: forall s t. KnownTyp t => KnownShape s => T s t -> Python DOC
generatePure x = do
  let sn = makeSn2 x
  mv <- snMapLookup2 sn <$> gets genPureTable
  case mv of
    Just v -> return v
    Nothing -> do
      e <- generatePure' (\s x' -> knownSShape s ?> generatePure x') typeSShape x
      v <- cache @s @t e
      modify (\g -> g {genPureTable = (snMapInsert2 sn v) (genPureTable g)})
      return v

genDistr :: forall s s0 t. KnownTyp t => Distribution s t -> SShape s0 -> SShape s -> DOC
genDistr d sh s1 = case d of
  TruncatedNormalD stddev -> funcall "tf.truncated_normal"
    [showSShape (sh .+. s1), named "stddev" (float stddev), named "dtype" (showTyp @t)]
  UniformD low high -> funcall "tf.random_uniform" [showSShape (sh .+. s1)
                                ,named "minval" (float low)
                                ,named "maxval" (float high)
                                ,named "dtype" (showTyp @t)]
  OrthogonalD ->
    funcall' (funcall "tf.orthogonal_initializer" [named "dtype" (showTyp @t)]) [named "shape" (showSShape (sh .+. s1))]

generatePure' :: forall s t. KnownTyp t => (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> Python DOC) -> SShape s -> T s t -> Python DOC
generatePure' rec sR = knownSShape sR ?> \case
  Unbroadcast{} -> error "broadcasting operation did not complete!"
  If c x y -> do
    rc <- rec typeSShape c
    rx <- rec typeSShape x
    ry <- rec typeSShape y
    return (func "tf.cond" [rc] [("true_fn", lambda0 rx) ,("false_fn", lambda0 ry) ,("strict","True")])
    where lambda0 z = text "lambda: " <> z
  -- if broadcast_to is broken: https://github.com/tensorflow/tensorflow/issues/21901
  -- DirectBroadcast s0 s1 s2 s3 x -> do
  --  recx <- rec (s0 .+. s2) x
  --  let expanded = func "tf.reshape" [recx,list (map (showDim' "-1")
  --         (concat [shapeToList' s0, genericReplicate (sListLength s1) 1
  --                 ,shapeToList' s2, genericReplicate (sListLength s3) 1 ]))] []
  --  return (funcall "tf.add" [expanded, func "tf.zeros" [showSShape sR] [("dtype", showTyp @t)]])
  DirectBroadcast s0 s1 s2 s3 x -> do
   recx <- rec (s0 .+. s2) x
   let expanded = func "tf.reshape" [recx,list (map (showDim' "-1")
          (concat [shapeToList' s0, genericReplicate (sListLength s1) 1
                  ,shapeToList' s2, genericReplicate (sListLength s3) 1 ]))] []
   return (funcall "tf.broadcast_to" [expanded, showSShape sR])
  Noise noiseId s0 s1 x -> do
    return $ (genDistr x s0 s1) <+> (text "# " <> integer noiseId)
  T op -> return $ case op of
    Variable v -> pyVarRepr v
    (Constant c) -> funcall "tf.constant" [pretty @t c, named "shape" (showSShape sR), named "dtype" (showTyp @t)]
    (Literal t) -> pyLitRepr t
    (Range n@Sat) -> (func "tf.range" [] [("start",integer 0),
                               ("limit",integer (natVal n)),
                               ("dtype",showTyp @t)])
  Where c x y -> do
    rc <- rec typeSShape c
    rx <- rec typeSShape x
    ry <- rec typeSShape y
    return (funcall "tf.where" [rc, rx, ry])
  UnOp operation s0  x -> do
   recx <- rec (s0 .+. unopInputShape operation) x
   return $ case operation of
    Diag _ -> funcall "tf.matrix_diag" [recx]
    Cast -> funcall "tf.cast" [recx,showTyp @t]
    StopGradient -> funcall "tf.stop_gradient" [recx]
    Axis1Op op' ->
       let (op,args) = case op' of
                         OneHot depth _ -> ("tf.one_hot",[("dtype",showTyp @t), ("depth", showDimS depth)])
                         ArgMax{} -> ("tf.argmax",[("output_type",showTyp @t)])
                         ReduceOp _ _ r -> ("tf.reduce_" ++ rop, [])
                            where rop = case r of
                                           Max -> "max"
                                           Min -> "min"
                                           Sum -> "sum"
                                           Mean -> "mean"
           axisName = if op == "tf.nn.softmax" then "dim" else "axis"  -- use dim before TF 1.5
       in func op [recx] ((axisName,integer (sListLength s0)):args)
    Float1Op op' -> funcall op (recx:args)
       where (op,args) = case op' of
                HardSigmoid -> ("tf.keras.backend.hard_sigmoid",[])
                Relu -> ("tf.nn.relu",[])
                ClipByValue lo hi -> ("tf.clip_by_value",[float lo,float hi])
                _ -> ("tf." ++ map toLower (show op'), [])
    Num1Op op' -> funcall op (recx:args)
       where (op,args) = case op' of
                Negate -> ("tf.negative",[])
                _ -> ("tf." ++ map toLower (show op'), [])
    SliceOp _ _ _ lo hi -> recx <> list (replicate (fromIntegral (sListLength s0)) (text ":") ++ [integer lo <> text ":" <> integer hi])
  MatMul s0 a b c x y  -> do
    recx <- rec (s0 .+. (:*) a ((:*) b Unit)) x
    recy <- rec (s0 .+. (:*) b ((:*) c Unit)) y
    return (funcall "tf.matmul" [recx, recy])
  BinOp operation s0 s1 _ s2 tt x y -> do
   recx <- rec (s0 .+. s1) x
   recy <- rec (s0 .+. s2) y
   return $ case operation of
     Simple2Op sop  -> let pop = case sop of
                                   Add -> "tf.add"
                                   Divide -> "tf.divide"
                                   Equal -> "tf.equal"
                                   Subtract -> "tf.subtract"
                                   Multiply -> "tf.multiply"
                                   Minimum -> "tf.minimum"
                                   Maximum -> "tf.maximum"
                                   LessThan -> "tf.less"
                       in funcall pop [recx,recy]
     SigmoidCrossEntropyWithLogits -> func "tf.nn.sigmoid_cross_entropy_with_logits" [] [("labels",recx),("logits",recy)]
     SparseSoftmaxCrossEntropyWithLogits -> func "tf.nn.sparse_softmax_cross_entropy_with_logits" []  [("labels",recx),("logits",recy)]
     SoftmaxCrossEntropyWithLogits -> func "tf.nn.softmax_cross_entropy_with_logits" []   [("labels",recx),("logits",recy)] -- FIXME: use _v2 for TF 1.5
  ReshapeFrom s t -> do
    rt <- rec s t
    return (funcall "tf.reshape" [rt, showShapeMinus sR])
  Concat s0 s1 xs -> do
    let go :: forall s0 s1 ns. SShape s0 -> SShape s1 -> NP (Catable s0 s1 t) ns -> Python [DOC]
        go _ _ Unit = return []
        go s0' s1' (Catable n y :* ys) = (:) <$> rec (s0' .+. n :* s1') y <*> go s0' s1' ys
    rxs <- go s0 s1 xs
    return (funcall "tf.concat" [list rxs, text "axis=" <> integer (sListLength s0)])
  Transpose s p x -> do
    rx <- rec s x
    return (func "tf.transpose" [rx] [("perm",list (map (integer . permToFun p) [0..sListLength s-1]))])
  Gather indexShape s0 m s1 x ix -> do
    rx <- rec (s0 .+. ((:*) m s1)) x
    rix <- rec indexShape ix
    return (func "tf.gather" [rx, rix] []) -- TODO this is generating something wrong...?
  GatherND containerShape elementShape indexShape x ix -> do
    rx <- rec (containerShape .+. elementShape) x
    rix <- rec (indexShape *: (sListLenAsNat containerShape)) ix
    return (func "tf.gather_nd" [rx, rix] [])
  Convolution bs inChans outChans filterShape s0 x filters -> do
    recx <- rec ((:*) bs (s0 *: inChans)) x
    recFilters <- rec (filterShape .+. ((:*) inChans ((:*) outChans Unit))) filters
    return (func "tf.nn.convolution" [recx, recFilters] [("padding",text (show ("SAME"::String))),("data_format", text (show dataFormat))])
   where dataFormat = case sListLength filterShape of
           1 -> ("NWC" :: String)
           2 -> "NHWC"
           3 -> "NDHWC"
           _ -> error "convolution: more than 3 spatial dimensions are not supported!"
  Pool bs window typ numChans outSpatial x -> do
     rx <- rec ((:*) bs (zipWithMulSShapes window outSpatial .+. (:*) numChans Unit)) x
     return (func "tf.nn.pool"
                  [rx, showSShape window, typ', text (show ("SAME" :: String))]
                  [("strides", showSShape window)])
   where typ' = text $ (show $ case typ of MaxPool -> "MAX"; AvgPool -> "AVG" :: String)
  Softmax _ _ x -> do
     rx <- rec typeSShape x
     return $ func "tf.nn.softmax" [rx] [("axis","1")]
  BatchNorm x -> do
     rx <- rec typeSShape x
     return $ func "tf.keras.layers.BatchNormalization(axis=1)" [rx] []

type Python a = StateT PyState (State GState) a

interpGen :: Gen a -> Python a
interpGen (GPReturn x) = return x
interpGen (GPBind a b) = do x <- interpGen a
                            interpGen (b x)
interpGen (GPVariable trainable name initial) = do
  i <- generatePure initial
  v <- newPyVar
  v <-- funcall "tf.Variable" [i, named "name" (string (show (name))), named "trainable" (bool trainable)]
  return v
interpGen (GPPlaceholder s t n) = do
  name <- newPyVar' s t
  name <-- funcall "tf.placeholder" [showSTyp t, named "shape" (showSShape s), named "name" (text (show n))]
  return name
interpGen (GPModify ref value) = do
  res <- newPyVar
  r <- generatePure ref
  v <- generatePure value
  res <-- (funcall "tf.assign" [r,v])
  return (T (Variable res))
interpGen (GPState f) = lift (state f)

-- TODO: get the parameters from the genParams field
-- | Return a list of parameters.
getParameters :: Python UntypedExpression
getParameters = return ("tf.trainable_variables()")

-- | Clip a gradient
clipByGlobalNorm :: Float -> UntypedExpression -> UntypedExpression
clipByGlobalNorm maxNorm x = funcall "tf.clip_by_global_norm" [x,float maxNorm] <> brackets (int 0)
 -- clip_by_global_norm returns a couple (clipped grads, global_norm)

-- | Gradient of wrt. given parameters.
grad :: UntypedExpression -> UntypedExpression -> UntypedExpression
grad y vars = funcall "tf.gradients" [y, vars]

-- | Batchify and compile a model with simple input to output mapping.
compile :: forall nAttrs ms batchSize sx tx sy ty sy_ ty_ p
        .  (KnownNat nAttrs, KnownShape ms, KnownNat batchSize, KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_, KnownTyp ty_, All KnownShape p, KnownLen p)
        => Options
        -> Gen (Tensor sx tx -> Tensor sy ty -> ModelOutputs ms ty_ p sy_)
        -> Python ()
compile options fGen = compileGen @nAttrs @ms @batchSize options (simpleModel <$> fGen)

-- | Batchify and compile a model with generic  input to output mapping and states
compileGen :: forall nAttrs ms batchSize shapesAndTypes sy_ ty_ ps stateShapes.
           (KnownShape ms, KnownNat batchSize, All KnownPlaceholder shapesAndTypes, KnownLen stateShapes,
            KnownLen shapesAndTypes, KnownNat nAttrs,
            All KnownShape stateShapes, KnownShape sy_, KnownTyp ty_, All KnownShape ps, KnownLen ps)
         => Options
         -> Gen (Placeholders shapesAndTypes -> HTV ty_ stateShapes -> StateAndOutputs ms ty_ ps (sy_ ': stateShapes))
         -> Python ()
compileGen options fGen =
  let batchedShapesKnown = mapFMap @(Cons batchSize) knownCons (allKnown @KnownShape @stateShapes typeSList)
  in knownAll batchedShapesKnown $
  -- TODO nAttrs is a nasty hack because we know sy == [batchSize, nAttrs]
     compileAlreadyBatched @batchSize options (precompile @batchSize @nAttrs (batchModel @batchSize fGen))

-- | Generic model compilation (do not use unless you know what you're doing)
compileAlreadyBatched :: forall bs ty stateShapes ps. KnownNat bs
           => KnownTyp ty
           => All KnownShape stateShapes
           => Options
           -> (Gen (NP (K ((String,HTV ty stateShapes,Scalar Float32))) ps)) -> Python ()
compileAlreadyBatched Options{..} models = do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    params <- getParameters
    let go :: forall xs. NP (K (String,HTV ty stateShapes,Scalar Float32)) xs -> Python [(String,DOC)]
        go Unit = return []
        go (K (prefix,updates,lossIn) :* ms) = do
          updates' <- untypedExprs updates
          loss <- generatePure lossIn
          trainStep <-
            if "test" `isPrefixOf` prefix
            then return (text "None") -- otherwise tensorflow complains. (This is for tasks which are not for training but testing only.)
            else assignAny $ case maxGradientNorm of
                               Nothing -> funcall "optimizer.minimize" [loss]
                               Just clip -> funcall "optimizer.apply_gradients"
                                              [funcall "zip" [clipByGlobalNorm clip (grad loss params),params]]
          let pks = map (first (prefix ++)) [("optimizer", (text "optimizer"))
                                            ,("params", params)
                                            ,("train", trainStep)
                                            ,("loss", loss)]
                    ++ [("update", list updates')]
          (pks ++) <$> go ms
    updatesAndLosses <- interpGen models
    peeks3 <- go updatesAndLosses
    peeks <- mapM paramToPeek =<< lift (gets genPeeks)
    let peeks2 = [("batch_size", (showDim @ bs))]
    gen (text "return " <> dict (peeks2 ++ peeks ++peeks3))

compileFn :: (All KnownPair ps, KnownShape s, KnownTyp t) => String -> HHTV ps -> Gen (T s t) -> Python ()
compileFn name args body = do
  arglist <- renderArgs args
  genFun name arglist $ do
    result <- interpGen body
    retval <- generatePure result
    gen (text "return " <> retval)

renderArgs :: All KnownPair xs => HHTV xs -> Python [DOC]
renderArgs Unit = return []
renderArgs (Uncurry x :* xs) = (:) <$> pure "_" <*> renderArgs xs

paramToPeek :: VarInfo -> Python (String,UntypedExpression)
paramToPeek (VarInfo name s t x) = do
  x' <- knownSShape s ?> (knownTyp t $ generatePure x)
  return (name,x')

untypedExprs :: All KnownShape xs => KnownTyp t =>  HTV t xs -> Python [DOC]
untypedExprs Unit = return []
untypedExprs (F x :* xs) = (:) <$> generatePure x <*> untypedExprs xs

-- | Pretty-print a Haskell type into a "DOC".
pretty :: forall t. KnownTyp t => HaskType t -> DOC
pretty = case kindVal @(TypKind t) of
  SInt -> case bitsVal @(TypBits t) of
    SB32 -> int . fromIntegral
    SB64 -> int . fromIntegral
  SBool -> bool
  SFloat -> case bitsVal @(TypBits t) of
    SB32 -> float
    SB64 -> double

-- | Python code of the runtime system.
rtsCode :: ByteString
rtsCode = $(embedFile "typedflow_rts.py")

-- | State of the Python code being generated.
data PyState = PyState {genText :: DOC
                       ,genPureTable :: SSNMap2 Shape Typ T DOC
                       -- ^ Table mapping pointers to their
                       -- interpretations, so that sharing in the data
                       -- structures can be exploited when generating
                       ,genAssignTable :: M.Map String DOC
                       -- ^ Table mapping expressions to variables, so
                       -- that lost sharing can be recovered
                       -- genPeeks :: [(String,UntypedExpression)]
                       }

type UntypedExpression = DOC
type DOC = Doc ()

