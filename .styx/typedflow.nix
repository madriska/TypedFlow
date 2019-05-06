{ mkDerivation, array, base, containers, ghc-typelits-knownnat
, hpack, mtl, pretty-compact, QuickCheck, stdenv
}:
mkDerivation {
  pname = "typedflow";
  version = "0.9";
  src = /home/bradediger/devel/archetyp/typedflow;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    base containers ghc-typelits-knownnat mtl pretty-compact
  ];
  libraryToolDepends = [ hpack ];
  executableHaskellDepends = [
    array base containers ghc-typelits-knownnat mtl pretty-compact
    QuickCheck
  ];
  preConfigure = "hpack";
  description = "Typed frontend to TensorFlow and higher-order deep learning";
  license = stdenv.lib.licenses.lgpl3;
}
