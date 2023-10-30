import torch.nn as nn
import torch
import torch.distributed as dist
import os, sys, time
""" Important summary of this case!!!
    After a layer generate an `output` during forward pass, `buf` = output.clone().detach() 
    will break the graph and start a new autograd graph which takes `buf` as inputs. When 
    backward, the new graph will stop at `buf`, and you should take (output, buf.grad) as 
    autograd.backward() input for the rest calculation.
    
    The performance comparation shows minor differences between partial and fully backward.
"""

def _allocate_buffer(shape, dtype):
    return torch.zeros(size=shape, device=0, dtype=dtype)

class Model0(nn.Module):
    def __init__(self, batch_size):
        self.bs = batch_size
        super(Model0, self).__init__()

        self.fc1 = nn.Linear(32, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc7a = nn.Linear(4096, 4096)
        self.fc7b = nn.Linear(4096, 4096)
        self.fc7c = nn.Linear(4096, 4096)
        self.loss = Loss(self.bs)


    def forward(self, x, is_break=False):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.fc3(o)
        o = self.fc4(o)
        o = self.fc5(o)
        o = self.fc6(o)
        o = self.fc7(o)
        if is_break:
            self.tmp_out7 = o
            self.buffer = o.clone().detach()
            self.buffer.requires_grad = True
            o = self.fc7a(self.buffer) # create new graph start with self.buffer
        else:
            o = self.fc7a(o)
        o = self.fc7b(o)
        o = self.fc7c(o)
        loss = self.loss(o)
        return loss


class Loss(nn.Module):
    def __init__(self, batch_size):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        # mock target value y_
        self.y_ = torch.ones((self.batch_size, 4096), device=0, dtype=torch.float32)
    
    def forward(self, x):
        loss = self.loss_fn(x, self.y_)
        return loss

loop = 100
batch_size = 100

model = Model0(batch_size=batch_size).cuda(0)

def backward_hook(module, inp_grad):
    print(f"Call module backward: {module}")
    pass
# for name, module in model.named_children():
#     module.register_full_backward_pre_hook(backward_hook)

x = torch.randn((batch_size, 32), dtype=torch.float32).cuda(0)
x.requires_grad = True

for i in range(50):
    #### Use two partial backward ###
    loss = model(x, is_break=True)
    # once backward
    t1_start = time.time()
    torch.autograd.backward(tensors=(loss, ))
    # print(model.buffer.grad) # except to have grad
    # print(model.tmp_out7.grad) # no grad
    # print(model.fc7.weight.grad) # no grad
    torch.autograd.backward(tensors=(model.tmp_out7, ), grad_tensors=model.buffer.grad)
    # print(model.fc7.weight.grad) # except to have grad
    # print(x.grad) # except to have grad
    t1_stop = time.time()
    print(f"Partial backward dur={t1_stop - t1_start}")

    ### Fully backward ###
    loss = model(x, is_break=False)
    # once backward
    t1_start = time.time()
    torch.autograd.backward(tensors=(loss, ))
    t1_stop = time.time()
    print(f"Fully backward dur={t1_stop - t1_start}")