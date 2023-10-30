import torch.nn as nn
import torch
import torch.distributed as dist
import os, sys, time
from open_clip.tprofiler import get_timers
timers = get_timers()

class Model0(nn.Module):
    def __init__(self):
        super(Model0, self).__init__()
        # 14 layers
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
        self.fc7d = nn.Linear(4096, 4096)
        self.fc7e = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        # self.fc8a = nn.Linear(4096, 4096)
        # self.fc8b = nn.Linear(4096, 4096)
        # self.fc8c = nn.Linear(4096, 4096)
        # self.fc8d = nn.Linear(4096, 4096)
        # self.fc8e = nn.Linear(4096, 4096)
        # self.fc8f = nn.Linear(4096, 4096)
        # self.fc8g = nn.Linear(4096, 4096)
        # self.fc8h = nn.Linear(4096, 4096)
        # self.fc9 = nn.Linear(4096, 4096)
        # self.fc10 = nn.Linear(4096, 4096)
        self.fc11 = nn.Linear(4096, 128)
        self.fc12 = nn.Linear(128, 32)

    def forward(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.fc3(o)
        o = self.fc4(o)
        o = self.fc5(o)
        o = self.fc6(o)
        o = self.fc7(o)
        o = self.fc7a(o)
        o = self.fc7b(o)
        o = self.fc7c(o)
        o = self.fc7d(o)
        o = self.fc7e(o)
        o = self.fc8(o)
        # o = self.fc8a(o)
        # o = self.fc8b(o)
        # o = self.fc8c(o)
        # o = self.fc8d(o)
        # o = self.fc8e(o)
        # o = self.fc8f(o)
        # o = self.fc8g(o)
        # o = self.fc8h(o)
        # o = self.fc9(o)
        # o = self.fc10(o)
        o = self.fc11(o)
        o = self.fc12(o)
        return o


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(32, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, 4096)
        self.fc10 = nn.Linear(4096, 4096)
        self.fc10a = nn.Linear(4096, 4096)
        # self.fc10b = nn.Linear(4096, 4096)
        self.fc11 = nn.Linear(4096, 128)
        self.fc12 = nn.Linear(128, 1)

    def forward(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.fc3(o)
        o = self.fc4(o)
        o = self.fc5(o)
        o = self.fc6(o)
        o = self.fc7(o)
        o = self.fc8(o)
        o = self.fc9(o)
        o = self.fc10(o)
        o = self.fc10a(o)
        # o = self.fc10b(o)
        o = self.fc11(o)
        o = self.fc12(o)
        return o
    
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(32, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc3a = nn.Linear(4096, 4096)
        self.fc3b = nn.Linear(4096, 4096)
        self.fc3c = nn.Linear(4096, 4096)
        self.fc3d = nn.Linear(4096, 4096)
        # self.fc3e = nn.Linear(4096, 4096)
        # self.fc3f = nn.Linear(4096, 4096)
        # self.fc3g = nn.Linear(4096, 4096)
        # self.fc3h = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.fc3(o)
        o = self.fc3a(o)
        o = self.fc3b(o)
        o = self.fc3c(o)
        o = self.fc3d(o)
        # o = self.fc3e(o)
        # o = self.fc3f(o)
        # o = self.fc3g(o)
        # o = self.fc3h(o)
        o = self.fc4(o)
        o = self.fc5(o)
        return o
    
def init_linear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(mean=0, std=0.01)
            m.bias.data.fill_(0.0)
    
class Loss(nn.Module):
    def __init__(self, batch_size):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        # mock target value y_
        self.y_ = torch.ones((self.batch_size, 1), device=local_rank, dtype=torch.float32)
    
    def forward(self, x0, x1):
        union_out = torch.add(x0, x1)
        loss = self.loss_fn(union_out, self.y_)
        return loss

loop = 100
batch_size = 100
local_rank = int(sys.argv[1])
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=local_rank, world_size=2)
if local_rank == 0:
    target_rank = 1
else:
    target_rank = 0

ng = dist.new_group([0, 1])
loop_end_time = time.time()

# TODO register for grad backward
if local_rank == 0:
    model = Model0().cuda(local_rank)
    
    def module_hook_activation_send(module, inp, out):
        print(f"Call stage0 send activation hook!, send shape={out.shape}")
        dist.send(out, module.target)
    
    def module_hook_grad_recv_hook(module, inp_grad):
        print(f"Call stage0 recv grad hook! Get mock in_grad={inp_grad}")
        dist.recv(module.recv_buffer, module.target)
        print(f"Recv real grad finish! grad={module.recv_buffer}")
    
    for name, module in model.named_children():
        if name == 'fc12':
            module.recv_buffer = torch.zeros((batch_size, 32), device=local_rank, dtype=torch.float32)
            module.target = target_rank
            module.register_forward_hook(module_hook_activation_send)
            module.register_full_backward_pre_hook(module_hook_grad_recv_hook)
            # module.register_full_backward_hook(grad_recv_post_hook)
            break

if local_rank == 1:
    model1 = Model1().cuda(local_rank)
    model2 = Model2().cuda(local_rank)
    loss = Loss(batch_size).cuda(local_rank)

    def module_hook_recv_activation(module, inp):
        print("Call stage1 recv activation hook!")
        dist.recv(module.recv_buffer, module.target)
        recvd = module.recv_buffer.clone().detach()
        recvd.requires_grad = recvd.is_floating_point()
        module.inputs_ = recvd
        module.inputs_.register_hook(tensor_hook_grad_send)
        return module.inputs_
    
    def tensor_hook_grad_send(grad):
        # Current send grad is not right, should send input_tensor.grad 
        print(f"Call stage1 send grad hook! grad={grad}")
        # target = 0 # TODO how make it a parameter?
        global target_rank
        dist.send(grad, target_rank)
        
    def module_hook_grad_send_hook(module, inp_grad):
        print(f"Call stage0 recv grad hook! Get mock in_grad={inp_grad}")
        dist.recv(module.recv_buffer, module.target)
        print(f"Recv real grad finish! grad={module.recv_buffer}")
    
    # alloc. buffers
    for name, module in model2.named_children():
        if name == 'fc1':
            module.recv_buffer = torch.zeros((batch_size, 32), device=local_rank, dtype=torch.float32)
            module.inputs_ = None
            module.target = target_rank
            module.register_forward_pre_hook(module_hook_recv_activation)
            # module.register_full_backward_hook(grad_send_post_hook)
            break
        
x = torch.ones((batch_size, 32)).cuda(local_rank)
    
for i in range(loop):
    if local_rank == 0:
        timers("rank0_all").start()
        # calc0
        timers("rank0_forward").start()
        o = model(x)
        timers("rank0_forward").stop()
        # send
        # timers("rank0_send").start()
        # dist.send(o, target_rank)
        # timers("rank0_send").stop()
        # dist.barrier(ng) # 增加这个barrier无影响，因为send/recv本身的依赖跟这个barrier效果相同
        
        timers("rank0_backward").start()
        torch.autograd.backward(tensors=(o, ), grad_tensors=(model.fc12.recv_buffer, ))
        timers("rank0_backward").stop()
        
        dist.barrier(ng)
        timers("rank0_all").stop()
        print(f"loop time:{time.time() - loop_end_time}")
        loop_end_time = time.time()
        
    else:
        timers("rank1_all").start()
    
        # calc1 
        timers("rank1_forward1").start()
        out1 = model1(x)
        timers("rank1_forward1").stop()
    
        # calc2
        timers("rank1_forward2").start()
        out2 = model2(None) # mock input
        timers("rank1_forward2").stop()
    
        # recv
        # timers("rank1_recv").start()
        # dist.recv(recv_buffer, target_rank)
        # timers("rank1_recv").stop()
        # dist.barrier(ng)
        
        # calc loss
        timers("rank2_loss").start()
        o = loss(out1, out2)
        timers("rank2_loss").stop()
        
        timers("rank2_backward").start()
        # (1) original backward (exec. forward latter, exec. backward earler.)
        o.backward()
        # (2) scheduled backward version
        # o.backward(inputs=(model2.fc1.inputs_), retain_graph=True)
        # o.backward()
        timers("rank2_backward").stop()
        
        # print(f"after stage1 backward, grad={model2.fc1.new_input.grad}")
        
        dist.barrier(ng)
        timers("rank1_all").stop()

    timers.log_all()
    
    
'''
for i in range(loop):
    if local_rank == 0:
        time.sleep(1)
        timers("send_all").start()
        timers("send").start()
        x = torch.zeros((10, 32), device=local_rank, dtype=torch.float32)
        dist.send(x, target_rank)
        timers("send").stop()
        dist.barrier()
        timers("send_all").stop()
    else:
        timers("recv_all").start()
        timers("recv").start()
        recv_buffer = torch.zeros((10, 32), device=local_rank, dtype=torch.float32)
        dist.recv(recv_buffer, target_rank)
        timers("recv").stop()
        dist.barrier()
        timers("recv_all").stop()
    timers.log_all()
        
'''