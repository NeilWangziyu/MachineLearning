import torch

@torch.jit.script
def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy

@torch.jit.script
def simple_lstm(input, hidden, wih, whh, bih, bhh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    outputs = []
    inputs = input.unbind(0)
    for seq_idx in range(len(inputs)):
        hidden = lstm_cell(inputs[seq_idx], hidden, wih, whh, bih, bhh)
        hy, _ = hidden
        outputs.append(hy)
    return hidden

print(simple_lstm.graph)
