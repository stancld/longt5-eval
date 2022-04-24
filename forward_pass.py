import argparse

import gin
import jax.numpy as jnp
import numpy as np
import torch

import t5x
from transformers import AutoModelForSeq2SeqLM, FlaxAutoModelForSeq2SeqLM


def main(config_file: str, checkpoint_dir: str, hf_model_path: str, run_torch: bool, seq_length: int) -> None:
    # Prepare input
    shape = [2, seq_length]
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
    try:
        embeddings = t5x_checkpoint["target"]["encoder"]["side_relpos_bias"]["rel_embedding"].T
        print("FlaxFormer global relpos:", embeddings.sum(), embeddings.shape)
    except:
        pass
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
    if run_torch:
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
    try:
        flax_embeddings = flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["TransientGlobalSelfAttention"][
            "global_relative_attention_bias"
        ]["embedding"]
        print("HF Flax global relpos:", flax_embeddings.sum(), flax_embeddings.shape)
    except:
        pass

    flax_output = flax_model(input_ids=encoder_input_tokens, decoder_input_ids=decoder_target_tokens).logits
    print(flax_output.shape)
    print("~~~~~~~~~~~~~~~~~~~~~~")

    ### Compare outputs ###
    print("FlaxFormer output:", output.sum())
    if run_torch:
        print("HF PyTorch output:", pt_output.sum())
    print("HF Flax output:", flax_output.sum())

    # Compare argmax
    print("FlaxFormer output:", jnp.argmax(output, axis=-1).sum())
    print("HF Flax output:", jnp.argmax(flax_output, axis=-1).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--hf-model-path")
    parser.add_argument("--run-torch", action="store_true")
    parser.add_argument("--seq-length", type=int, default=10)
    args = parser.parse_args()

    main(args.config_file, args.checkpoint_dir, args.hf_model_path, args.run_torch, args.seq_length)
