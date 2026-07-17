


def forward(self, x : _torch_Tensor_) -> _torch_Tensor_:
    gate_up_proj = self.gate_up_proj(x);  x = None
    chunk = torch.chunk(gate_up_proj, 2, dim = -1);  gate_up_proj = None
    getitem = chunk[0]
    getitem_1 = chunk[1];  chunk = None
    sigmoid = torch.sigmoid(getitem)
    mul = getitem * sigmoid;  getitem = sigmoid = None
    mul_1 = mul * getitem_1;  mul = getitem_1 = None
    down_proj = self.down_proj(mul_1);  mul_1 = None
    return down_proj
    