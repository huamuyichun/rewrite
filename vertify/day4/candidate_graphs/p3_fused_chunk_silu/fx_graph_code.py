


def forward(self, x : _torch_Tensor_) -> _torch_Tensor_:
    gate_up_proj = self.gate_up_proj(x);  x = None
    chunk = torch.chunk(gate_up_proj, 2, dim = -1);  gate_up_proj = None
    getitem = chunk[0]
    getitem_1 = chunk[1];  chunk = None
    silu = torch.nn.functional.silu(getitem, inplace = False);  getitem = None
    mul = silu * getitem_1;  silu = getitem_1 = None
    down_proj = self.down_proj(mul);  mul = None
    return down_proj
    