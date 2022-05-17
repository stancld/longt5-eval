# ONNX export

[v4] Last, we also need to replace `x.T` with `x.transpose(0, 1)` operation to make sure everything works just ok.

[v3] It looks like `torch.tile(...)` can be replaced with `x.repeat(...)` in order to enable onnx conversion.

[v2]
It was found out it is relatively simple to substitute `torch.maximum/minimum` operation with
`torch.where`. ~~Nonetheles, another problem occurs with `torch.tile` method.~~

[v1]
Currently, ONNX conversion works fine for the model with a local attention with `opset_version>=11`. ~~However, ONNX export is not supported for a transient-global model as exportin `torch.maximum`
operation is not supported yet.~~
