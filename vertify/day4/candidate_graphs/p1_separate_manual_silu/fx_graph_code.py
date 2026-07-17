


def forward(self, x : _torch_Tensor_) -> _torch_Tensor_:
    gate_proj = self.gate_proj(x)
    up_proj = self.up_proj(x);  x = None
    sigmoid = torch.sigmoid(gate_proj)
    mul = gate_proj * sigmoid;  gate_proj = sigmoid = None
    mul_1 = mul * up_proj;  mul = up_proj = None
    down_proj = self.down_proj(mul_1);  mul_1 = None
    return down_proj
    