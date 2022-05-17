# ONNX export

[v2]
It was found out it is relatively simple to substitute `torch.maximum/minimum` operation with
`torch.where`. Nonetheles, another problem occurs with `torch.tile` method.

[v1]
Currently, ONNX conversion works fine for the model with a local attention with `opset_version>=11`. ~~However, ONNX export is not supported for a transient-global model as exportin `torch.maximum`
operation is not supported yet.~~
