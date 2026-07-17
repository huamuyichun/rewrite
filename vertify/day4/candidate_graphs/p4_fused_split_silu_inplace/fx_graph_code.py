


def forward(self, x : _torch_Tensor_) -> _torch_Tensor_:
    gate_up_proj = self.gate_up_proj(x);  x = None
    split = torch.functional.split(gate_up_proj, 4096, dim = -1);  gate_up_proj = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    silu = torch.nn.functional.silu(getitem, inplace = False);  getitem = None
    mul_ = silu.mul_(getitem_1);  silu = getitem_1 = None
    down_proj = self.down_proj(mul_);  mul_ = None
    return down_proj
    