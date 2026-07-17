


def forward(self, x : _torch_Tensor_) -> _torch_Tensor_:
    gate_proj = self.gate_proj(x)
    up_proj = self.up_proj(x);  x = None
    silu = torch.nn.functional.silu(gate_proj, inplace = False);  gate_proj = None
    mul = silu * up_proj;  silu = up_proj = None
    down_proj = self.down_proj(mul);  mul = None
    return down_proj
    