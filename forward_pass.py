import argparse

import gin
import numpy as np
import torch

import t5x
from transformers import AutoModelForSeq2SeqLM, FlaxAutoModelForSeq2SeqLM


def main(config_file: str, checkpoint_dir: str, hf_model_path: str) -> None:
    # Prepare input
    shape = [2, 10]
    encoder_input_tokens = np.ones(shape, dtype=np.int32)
    decoder_input_tokens = np.ones(shape, dtype=np.int32)
    decoder_target_tokens = np.ones(shape, dtype=np.int32)

    ################
    ## FlaxFormer ##
    ################

    # Parse config file
    gin.parse_config_file(config_file)
    gin.finalize()

    # Get model
    model_config_ref = gin.query_parameter("%MODEL")
    model = model_config_ref.scoped_configurable_fn()

    # Load checkpoint
    t5x_checkpoint = t5x.checkpoints.load_t5x_checkpoint(checkpoint_dir)

    # Run forward pass
    print("~~~~~~~~~~ FlaxForrmer ~~~~~~~~~~~~")
    output = model.module.apply(
        {"params": t5x_checkpoint["target"]},
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=decoder_input_tokens,
        decoder_target_tokens=decoder_target_tokens,
        enable_dropout=False,
    )

    # Print output shape
    print(output.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~")

    #################
    ## HuggingFace ##
    #################
    pt_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_path)
    print("~~~~~~~~~ HF PyTorch ~~~~~~~~~~~~~")
    with torch.no_grad():
        pt_output = pt_model(
            input_ids=torch.from_numpy(encoder_input_tokens).long(),
            labels=torch.from_numpy(decoder_target_tokens).long(),
        ).logits

    print(pt_output.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~")

    flax_model = FlaxAutoModelForSeq2SeqLM.from_pretrained(hf_model_path)
    print("~~~~~~~~~ HF Flax ~~~~~~~~~~~~~")
    flax_output = flax_model(input_ids=encoder_input_tokens, decoder_input_ids=decoder_target_tokens).logits
    print(flax_output.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~")

    ### Compare outputs ###
    print("FlaxFormer output:", output.sum())
    print("HF PyTorch output:", pt_output.sum())
    print("HF Flax output:", flax_output.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--hf-model-path")
    args = parser.parse_args()

    main(args.config_file, args.checkpoint_dir, args.hf_model_path)
