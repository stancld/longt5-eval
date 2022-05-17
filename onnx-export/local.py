import tempfile

import torch
from transformers import LongT5Model, LongT5Config

OPSET_VERS = range(9, 16)

if __name__ == "__main__":
    config = LongT5Config()
    model = LongT5Model(config)
    for opset_version in OPSET_VERS:
        print(f"Running ONNX conversion with opset version = {opset_version} ...")
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                torch.onnx.export(
                    model,
                    (torch.arange(0, 10).reshape(1, -1), torch.ones(1, 10), torch.arange(0, 10).reshape(1, -1)),
                    f"{tmpdirname}/longt5_test.onnx",
                    export_params=True,
                    opset_version=opset_version,
                    input_names=["input_ids", "decoder_input_ids"],
                )
            except RuntimeError as e:
                print(f"Opset version = {opset_version} :: {e!r}")
