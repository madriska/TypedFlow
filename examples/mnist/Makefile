test: mnist_model.py main.py
	nix-shell ../seq2seq/shell.nix --run "python main.py"
# TODO ^^ fix the shell.nix reference to be somewhere more centralized

mnist_model.py: MNIST.hs
	nix-shell ../../.styx/shell.nix --run "ghci -i../.. MNIST.hs -e main"

