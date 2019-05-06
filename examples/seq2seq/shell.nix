{ nixpkgs ? import <nixpkgs> {} }:
let
  # nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/4cf0b6ba5d5ab5eb20a88449e0612f4dad8e4c29.tar.gz;
  # nixpkgs_source = fetchTarball https://releases.nixos.org/nixos/19.03/nixos-19.03.172470.a177da6b86e/nixexprs.tar.xz;
  nixpkgs_source = fetchTarball https://releases.nixos.org/nixos/unstable/nixos-19.09pre178539.d457e31d9ec/nixexprs.tar.xz;
in with (import nixpkgs_source {}).pkgs;
let py = (pkgs.python36.withPackages (ps: [ps.tensorflowWithCuda ps.nltk]));

in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ py ];
}

