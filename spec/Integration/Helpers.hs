module Integration.Helpers where

import Data.Text (Text)
import qualified Data.Text.IO as T
import Language.Haskell.TH
import Language.Haskell.TH.Quote
import qualified NeatInterpolation
import System.Process (readProcess)

runPython :: String -> Text -> IO String
runPython moduleName python = do
  let path = "spec/generated/" <> moduleName <> ".py"
  T.writeFile path python

  readProcess "/usr/bin/python3" ["-m", "spec.generated." <> moduleName] ""
  
-- | Quasiquoter to dedent literal Python.
py :: QuasiQuoter
py = NeatInterpolation.text

