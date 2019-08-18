import torch
import torchvision
#
# dummy_input = torch.randn(10, 3, 224, 224)
# model = torchvision.models.alexnet(pretrained=True)
#
# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]
#
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

# ----------------

# Trace-based only

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        for i in range(y):
            x = x + i
        return x

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)


# Mixing tracing and scripting

@torch.jit.script
def loop(x, y):
    for i in range(int(y)):
        x = x + i
    return x

class LoopModel2(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

model = LoopModel2()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                  input_names=['input_data', 'loop_range'])