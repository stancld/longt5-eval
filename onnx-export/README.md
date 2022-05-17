# ONNX export

Currently, ONNX conversion works fine for the model with a local attention with `opset_version>=11`. However, ONNX
export is not supported for a transient-global model as exportin `torch.maximum` operation is not supported yet.