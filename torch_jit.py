import torch
from typing import Tuple, List, Dict

def my_function(x):
    return x * 2

ftrace = torch.jit.trace(my_function, (torch.ones(2,2)))

y = torch.ones(2,2).add_(1.0)

print(ftrace.graph)
print(ftrace(y))
print(type(ftrace))
# ftrace.save("ftrace.pt")

@torch.jit.script
def my_function(x):
    #if x.mean() > 1.0:
    if bool(x.mean() > 1.0):
        #r = torch.tensor(1.0)
        r = 1
    else:
        #r = torch.tensor(2.0)
        r = 2
    return r

y = torch.ones(2,2).add_(1.0)
print(my_function.graph)
print(my_function(y))