import torch
from typing import Tuple, List, Dict

def my_function(x):
    return x * 2

ftrace = torch.jit.trace(my_function, (torch.ones(2,2)))

y = torch.ones(2,2).add_(1.0)

print(ftrace.graph)
print(ftrace(y))
print(type(ftrace))
ftrace.save("ftrace.pt")



y = torch.ones(2,2).add_(1.0)
print(my_function(y))

class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method
    def forward(self, input):
        if bool(input.sum() > 0):
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_script_module = MyModule(2, 3)

my_script_module.save("model.pt")

@torch.jit.script
def foo(x:int, y:int)->int:
    if x > y:
        r = x
    else:
        r = y
    return r

foo.save("foo.pt")


foo_new = torch.jit.load('foo.pt')
res1 = foo(10,9)
res2 = foo_new(9, 10)
print(res1, res2)
print(foo.graph)

