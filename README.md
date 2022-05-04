# LongT5x -> HF transformers

This is an helper repo for the verification of the implementation equivalence during porting LongT5 model into the `transformers` package.

<hr>

## How to use the repo.

1. Run `setup.sh` file. This script downloads some original LongT5 checkpoints, and takes care of some necessary
Google packages like `T5x, Flaxformer`.

2. Then, there is an example python file `forward_pass` which contains logic for the evaluation both of the original model and
proposed HF-implemented model. For this purpose, there are some example bash script in `runnin_scripts/` folder.