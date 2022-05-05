# LongT5x -> HF transformers

This is an helper repo for the verification of the implementation equivalence during porting LongT5 model into the `transformers` package.

<hr>

## How to use the repo.

1. Run `setup.sh` file. This script downloads some original LongT5 checkpoints, and takes care of some necessary
Google packages like `T5x, Flaxformer`.

2. Then, there is an example python file `forward_pass` which contains logic for the evaluation both of the original model and
proposed HF-implemented model. For this purpose, there are some example bash script in `runnin_scripts/` folder.

## Current issue

Currently, there's a problem with TGlobal model where a discrepancy occurs when the sequence length is larger than `global_block_size`.
This can be illustrated by running (1) `longt5-tglobal_forward_pass.sh` , and (2) `longt5-tglobal_forward_pass_seq25.sh` scripts
in the `running_scripts/` folder. There is no issue with the output of the (1), however (2) does not work as expected. 
