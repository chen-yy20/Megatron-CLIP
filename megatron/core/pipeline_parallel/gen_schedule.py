from enum import Enum, auto
from typing import List

class Ops(Enum):
    forward = auto()
    backward = auto()
    send_forward = auto()
    recv_forward = auto()
    send_backward = auto()
    recv_backward = auto()
    send_forward_recv_backward = auto()
    send_backward_recv_forward = auto()
    synchronize = auto()
    
class PipelineInstruction():
    def __init__(self, op_type, model_chunk_id=None, micro_batch_id=None, up_or_down=None) -> None:
        self.op_type = op_type
        self.model_chunk_id = model_chunk_id
        if isinstance(micro_batch_id, tuple):
            self.micro_batch_id, self.recv_micro_batch_id = micro_batch_id
        else:
            self.micro_batch_id = micro_batch_id
        self.up_or_down = up_or_down
    
    def __str__(self) -> str:
        return "{}, model:{}, micro_batch_id:{}, pipeline direction:{}" \
                .format(self.op_type, self.model_chunk_id, self.micro_batch_id, self.up_or_down)

class Schedules1:
    def __init__(self) -> None:
        # Save ordered instructions
        self._inst = []
    
    def generate(self, rank, pipeline_parallel_world_size, num_microbatches, dual_model, modal_ratio=2) -> List[Ops]:
        """Computation schedules of bidirectional pipeline cannot be launched in order,
        because the opposite send ops will block the schedule. We rearrange the order
        to make the op list can execute one-by-one.
        
        Bidirectional pipeline impl. The model chuck order is [image, text].
        """

        num_warmup_microbatches = pipeline_parallel_world_size - rank - 1
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        if num_warmup_microbatches == 0:
            up_pipeline_warmup_batches = 0 # set for last vision stage
        else:
            up_pipeline_warmup_batches = min(rank * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - (rank + 1))
            if up_pipeline_warmup_batches < 0:
                up_pipeline_warmup_batches = 0
        up_pipeline_remaining_batches = num_microbatches - up_pipeline_warmup_batches
        if rank > 1:
            up_pipeline_next_stage_warmup_batches = min((rank -  1) * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - rank)
            if up_pipeline_next_stage_warmup_batches < 0:
                up_pipeline_next_stage_warmup_batches = 0
        else:
            up_pipeline_next_stage_warmup_batches = 0
        # print(f"warmup forward send, first round={up_pipeline_next_stage_warmup_batches}")
        
        # print(f"up pipeline warmup batch num={up_pipeline_warmup_batches}, remaining batch num={up_pipeline_remaining_batches}")
        send_up_pended = []
        send_down_pended = []
        # for text warmup
        if dual_model:
            for i in range(up_pipeline_warmup_batches):
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, i, "up"))
                self._inst.append(PipelineInstruction(Ops.forward, 1, i, "up"))
                pp_inst = PipelineInstruction(Ops.send_forward, 1, i, "up")
                if i < up_pipeline_next_stage_warmup_batches:
                    self._inst.append(pp_inst)
                else:
                    send_up_pended.append(pp_inst)
        # print(f"send_up_pended after text warmup={send_up_pended}")
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, i, "down"))
            
            # Run vision warmup forward    
            self._inst.append(PipelineInstruction(Ops.forward, 0, i, "down"))
            # last warmup batch in second last stage directly send, else pending
            pp_inst = PipelineInstruction(Ops.send_forward, 0, i, "down")
            if i == num_warmup_microbatches - 1:
                send_down_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
             
            # If is the last warmup forward, send up pipeline's out tensors.
            if i == num_warmup_microbatches - 1:
                if len(send_up_pended) > 0:
                    self._inst.append(send_up_pended.pop(0))
        
        set_manual_send_down = {0: 3, 1: 3, 2: 3, 3: -1}
        early_backward_batches = [0, 1] # TODO auto determine
        early_backward_stages = [0, 1] # TODO auto determine
        if rank < 2:
            early_backward_size = len(early_backward_batches)
        else:
            early_backward_size = 0
        early_backward_batch_index = []
        if rank == 0:
            early_backward_batch_index = [0, 1]
        elif rank == 1:
            early_backward_batch_index = [3, 3]
        if len(early_backward_batch_index) > 0:
            insert_place = early_backward_batch_index.pop(0)
        else:
            insert_place = -1
        # set_manual_send_down = {0: 7, 1: 7, 2: 4, 3: -1} # for vision forward, rank i: up microbatch id when sent
        early_recvd = []
        send_up_backward_pend = []
        # Run up pipeline remaining forward
        for i in range(up_pipeline_remaining_batches):
            micro_batch_id = i + up_pipeline_warmup_batches
            if micro_batch_id not in early_recvd:
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, micro_batch_id, "up"))
                if len(send_up_backward_pend) > 0:
                    self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.forward, 1, micro_batch_id, "up"))
            pp_inst = PipelineInstruction(Ops.send_forward, 1, micro_batch_id, "up")
            # Send pended up forward first
            if rank == 1 and micro_batch_id in [1, 2]:
                send_up_pended.append(pp_inst) # TODO how to handle?
                continue
            if len(send_up_pended) > 0:
                self._inst.append(send_up_pended.pop(0))
                send_up_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
            # Insert down forward manually
            if set_manual_send_down[rank] == micro_batch_id:
                # Insert this send down op means previous stage' forward have finished, so recvive all
                # remaining up pipeline's forward output and skip recv at the loop begaining.
                for j in range(i + 1, up_pipeline_remaining_batches):
                    micro_batch_id = j + up_pipeline_warmup_batches
                    self._inst.append(PipelineInstruction(Ops.recv_forward, 1, micro_batch_id, "up"))
                    early_recvd.append(micro_batch_id)
                self._inst.append(send_down_pended.pop(0))
            # Early backward
            if rank in early_backward_stages and micro_batch_id == insert_place:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                if rank == early_backward_stages[-1]:
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                    self._inst.append(send_up_pended.pop(0))
                else:
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                if len(early_backward_batch_index) > 0:
                    insert_place = early_backward_batch_index.pop(0)
                else:
                    insert_place == -1
        if rank in early_backward_stages:
            while len(early_backward_batches) > 0:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                if rank == early_backward_stages[-1]:
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                    self._inst.append(send_up_pended.pop(0))
                else:
                    self._inst.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                if len(early_backward_batch_index) > 0:
                    insert_place = early_backward_batch_index.pop(0)
                else:
                    insert_place == -1
                
        while len(send_up_pended) > 0:
            self._inst.append(send_up_pended.pop(0))
                
        if num_microbatches_remaining > 0:
            # recv the first steady phase vision forward
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, num_warmup_microbatches, "down"))

        # Run 1F1B in steady state.
        vision_backward_micro_batch_id = 0
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            micro_batch_id = num_warmup_microbatches + i
            self._inst.append(PipelineInstruction(Ops.forward, 0, micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_forward_recv_backward, 0, (micro_batch_id, vision_backward_micro_batch_id), "down"))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            if last_iteration:
                self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            else:
                self._inst.append(PipelineInstruction(Ops.send_backward_recv_forward, 0, (vision_backward_micro_batch_id, micro_batch_id + 1), "down"))
            vision_backward_micro_batch_id += 1
        
        # text cool down
        # set_manual_send_up = {0: -1, 1: 3, 2: 3, 3: 3} # send for vision backward after recv
        
        text_backward_micro_batch_id = 0 + early_backward_size        
        # Cool down
        for i in range(num_warmup_microbatches):
            # Backward text cool down
            if text_backward_micro_batch_id < num_microbatches:
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
                text_backward_micro_batch_id += 1
            # Backward vision cool down
            self._inst.append(PipelineInstruction(Ops.recv_backward, 0, vision_backward_micro_batch_id, "down"))
            if len(send_up_backward_pend) > 0:
                self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            vision_backward_micro_batch_id += 1
        
        # remaining text cool down
        while len(send_up_backward_pend) > 0:
            self._inst.append(send_up_backward_pend.pop(0))
        while text_backward_micro_batch_id < num_microbatches:
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
            text_backward_micro_batch_id += 1
            
        return self._inst
            
    def format_schedule(self):
        for sche in self._inst:
            vision_or_text = "vision" if sche.model_chunk_id == 0 else "text"
            if hasattr(sche, "recv_micro_batch_id"):
                recv_micro_batch_id = sche.recv_micro_batch_id
                print(f"{vision_or_text}, {sche.op_type}, send micro_batch_id={sche.micro_batch_id}, recv micro_batch_id={recv_micro_batch_id}")
            else:    
                print(f"{vision_or_text}, {sche.op_type}, micro_batch_id={sche.micro_batch_id}")



class Schedules2:
    def __init__(self) -> None:
        # Save ordered instructions
        self._inst = []
    
    def generate(self, rank, pipeline_parallel_world_size, num_microbatches, dual_model, modal_ratio=2) -> List[Ops]:
        """Computation schedules of bidirectional pipeline cannot be launched in order,
        because the opposite send ops will block the schedule. We rearrange the order
        to make the op list can execute one-by-one.
        
        Bidirectional pipeline impl. The model chuck order is [image, text].
        """

        num_warmup_microbatches = pipeline_parallel_world_size - rank - 1
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        if num_warmup_microbatches == 0:
            up_pipeline_warmup_batches = 0 # set for last vision stage
        else:
            up_pipeline_warmup_batches = min(rank * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - (rank + 1))
            if up_pipeline_warmup_batches < 0:
                up_pipeline_warmup_batches = 0
        up_pipeline_remaining_batches = num_microbatches - up_pipeline_warmup_batches
        if rank > 1:
            up_pipeline_next_stage_warmup_batches = min((rank -  1) * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - rank)
            if up_pipeline_next_stage_warmup_batches < 0:
                up_pipeline_next_stage_warmup_batches = 0
        else:
            up_pipeline_next_stage_warmup_batches = 0
        # print(f"warmup forward send, first round={up_pipeline_next_stage_warmup_batches}")
        
        # print(f"up pipeline warmup batch num={up_pipeline_warmup_batches}, remaining batch num={up_pipeline_remaining_batches}")
        send_up_pended = []
        send_down_pended = []
        # for text warmup
        if dual_model:
            for i in range(up_pipeline_warmup_batches):
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, i, "up"))
                self._inst.append(PipelineInstruction(Ops.forward, 1, i, "up"))
                pp_inst = PipelineInstruction(Ops.send_forward, 1, i, "up")
                if i < up_pipeline_next_stage_warmup_batches:
                    self._inst.append(pp_inst)
                else:
                    send_up_pended.append(pp_inst)
        # print(f"send_up_pended after text warmup={send_up_pended}")
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, i, "down"))
            
            # Run vision warmup forward    
            self._inst.append(PipelineInstruction(Ops.forward, 0, i, "down"))
            # last warmup batch in second last stage directly send, else pending
            pp_inst = PipelineInstruction(Ops.send_forward, 0, i, "down")
            if i == num_warmup_microbatches - 1:
                send_down_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
             
            # If is the last warmup forward, send up pipeline's out tensors.
            if i == num_warmup_microbatches - 1:
                if len(send_up_pended) > 0:
                    self._inst.append(send_up_pended.pop(0))
        
        set_manual_send_down = {0: 3, 1: 3, 2: 3, 3: -1}
        if rank == 0:
            early_backward_batches = [0, 1]
        elif rank == 1 or rank == 2:
            early_backward_batches = [0]
        else:
            early_backward_batches = []
        early_backward_stages = [0, 1, 2] # TODO auto determine
        early_backward_size = len(early_backward_batches)

        if rank == 0:
            early_backward_batch_index = [0, 3]
        elif rank == 1:
            early_backward_batch_index = [3]
        else:
            early_backward_batch_index = []
        if len(early_backward_batch_index) > 0:
            insert_place = early_backward_batch_index.pop(0)
        else:
            insert_place = -1
        early_recvd = []
        send_up_backward_pend = []
        # Run up pipeline remaining forward
        for i in range(up_pipeline_remaining_batches):
            micro_batch_id = i + up_pipeline_warmup_batches
            if micro_batch_id not in early_recvd:
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, micro_batch_id, "up"))
                if len(send_up_backward_pend) > 0:
                    self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.forward, 1, micro_batch_id, "up"))
            pp_inst = PipelineInstruction(Ops.send_forward, 1, micro_batch_id, "up")
            # pend sending because rank0 is doing backward
            if rank == 1 and micro_batch_id in [1, 2]:
                send_up_pended.append(pp_inst) # TODO how to handle?
                continue
            # Send pended up forward first
            if len(send_up_pended) > 0:
                self._inst.append(send_up_pended.pop(0))
                send_up_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
            # Insert send forward down manually
            if set_manual_send_down[rank] == micro_batch_id:           
                self._inst.append(send_down_pended.pop(0))
            # Insert early backward
            if rank in early_backward_stages and micro_batch_id == insert_place:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                if rank == 1:
                    self._inst.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                else:
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                if len(early_backward_batch_index) > 0:
                    insert_place = early_backward_batch_index.pop(0)
                else:
                    insert_place == -1
                
        while len(send_up_pended) > 0:
            self._inst.append(send_up_pended.pop(0))
        
        if num_microbatches_remaining > 0:
            # recv the first steady phase vision forward
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, num_warmup_microbatches, "down"))
        # Run 1F1B in steady state.
        vision_backward_micro_batch_id = 0
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            micro_batch_id = num_warmup_microbatches + i
            self._inst.append(PipelineInstruction(Ops.forward, 0, micro_batch_id, "down"))
            
            if rank == 2 and micro_batch_id == 1:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
            
            self._inst.append(PipelineInstruction(Ops.send_forward_recv_backward, 0, (micro_batch_id, vision_backward_micro_batch_id), "down"))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            if last_iteration:
                self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            else:
                self._inst.append(PipelineInstruction(Ops.send_backward_recv_forward, 0, (vision_backward_micro_batch_id, micro_batch_id + 1), "down"))
            vision_backward_micro_batch_id += 1
        
        # text cool down
        text_backward_micro_batch_id = 0 + early_backward_size        
        # Cool down
        for i in range(num_warmup_microbatches):
            # Backward text cool down
            if text_backward_micro_batch_id < num_microbatches:
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
                text_backward_micro_batch_id += 1
            # Backward vision cool down
            self._inst.append(PipelineInstruction(Ops.recv_backward, 0, vision_backward_micro_batch_id, "down"))
            if len(send_up_backward_pend) > 0:
                self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            vision_backward_micro_batch_id += 1
        
        # remaining text cool down
        while len(send_up_backward_pend) > 0:
            self._inst.append(send_up_backward_pend.pop(0))
        while text_backward_micro_batch_id < num_microbatches:
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
            text_backward_micro_batch_id += 1
            
        return self._inst
            
    def format_schedule(self):
        for sche in self._inst:
            vision_or_text = "vision" if sche.model_chunk_id == 0 else "text"
            if hasattr(sche, "recv_micro_batch_id"):
                recv_micro_batch_id = sche.recv_micro_batch_id
                print(f"{vision_or_text}, {sche.op_type}, send micro_batch_id={sche.micro_batch_id}, recv micro_batch_id={recv_micro_batch_id}")
            else:    
                print(f"{vision_or_text}, {sche.op_type}, micro_batch_id={sche.micro_batch_id}")

class Schedules3:
    def __init__(self) -> None:
        # Save ordered instructions
        self._inst = []
    
    def generate(self, rank, pipeline_parallel_world_size, num_microbatches, dual_model, modal_ratio=2) -> List[Ops]:
        """Computation schedules of bidirectional pipeline cannot be launched in order,
        because the opposite send ops will block the schedule. We rearrange the order
        to make the op list can execute one-by-one.
        
        Bidirectional pipeline impl. The model chuck order is [image, text].
        """

        num_warmup_microbatches = pipeline_parallel_world_size - rank - 1
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        if num_warmup_microbatches == 0:
            up_pipeline_warmup_batches = 0 # set for last vision stage
        else:
            up_pipeline_warmup_batches = min(rank * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - (rank + 1))
            if up_pipeline_warmup_batches < 0:
                up_pipeline_warmup_batches = 0
        up_pipeline_remaining_batches = num_microbatches - up_pipeline_warmup_batches
        if rank > 1:
            up_pipeline_next_stage_warmup_batches = min((rank -  1) * modal_ratio, num_microbatches) - \
                    (pipeline_parallel_world_size - rank)
            if up_pipeline_next_stage_warmup_batches < 0:
                up_pipeline_next_stage_warmup_batches = 0
        else:
            up_pipeline_next_stage_warmup_batches = 0
        # print(f"warmup forward send, first round={up_pipeline_next_stage_warmup_batches}")
        
        # print(f"up pipeline warmup batch num={up_pipeline_warmup_batches}, remaining batch num={up_pipeline_remaining_batches}")
        send_up_pended = []
        send_down_pended = []
        # for text warmup
        if dual_model:
            for i in range(up_pipeline_warmup_batches):
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, i, "up"))
                self._inst.append(PipelineInstruction(Ops.forward, 1, i, "up"))
                pp_inst = PipelineInstruction(Ops.send_forward, 1, i, "up")
                if i < up_pipeline_next_stage_warmup_batches:
                    self._inst.append(pp_inst)
                else:
                    send_up_pended.append(pp_inst)
        # print(f"send_up_pended after text warmup={send_up_pended}")
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, i, "down"))
            
            # Run vision warmup forward    
            self._inst.append(PipelineInstruction(Ops.forward, 0, i, "down"))
            # last warmup batch in second last stage directly send, else pending
            pp_inst = PipelineInstruction(Ops.send_forward, 0, i, "down")
            if i == num_warmup_microbatches - 1:
                send_down_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
             
            # If is the last warmup forward, send up pipeline's out tensors.
            if i == num_warmup_microbatches - 1:
                if len(send_up_pended) > 0:
                    self._inst.append(send_up_pended.pop(0))
        
        set_manual_send_down = {0: 3, 1: 3, 2: 3, 3: -1}
        
        early_backward_batches = [0, 1]
        early_backward_stages = [0, 1, 2, 3] # TODO auto determine
        early_backward_size = len(early_backward_batches)

        if rank == 0:
            early_backward_batch_index = [0, 3]
        elif rank == 1:
            early_backward_batch_index = [3]
        else:
            early_backward_batch_index = []
        if len(early_backward_batch_index) > 0:
            insert_place = early_backward_batch_index.pop(0)
        else:
            insert_place = -1
        early_recvd = []
        send_up_backward_pend = []
        # Run up pipeline remaining forward
        for i in range(up_pipeline_remaining_batches):
            micro_batch_id = i + up_pipeline_warmup_batches
            if micro_batch_id not in early_recvd:
                self._inst.append(PipelineInstruction(Ops.recv_forward, 1, micro_batch_id, "up"))
                if len(send_up_backward_pend) > 0:
                    self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.forward, 1, micro_batch_id, "up"))
            pp_inst = PipelineInstruction(Ops.send_forward, 1, micro_batch_id, "up")
            # pend sending because rank0 is doing backward
            if rank == 1 and micro_batch_id in [1, 2]:
                send_up_pended.append(pp_inst) # TODO how to handle?
                continue
            # Send pended up forward first
            if len(send_up_pended) > 0:
                self._inst.append(send_up_pended.pop(0))
                send_up_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
            # Insert send forward down manually
            if set_manual_send_down[rank] == micro_batch_id:           
                self._inst.append(send_down_pended.pop(0))
            # Insert early backward
            if rank in early_backward_stages and micro_batch_id == insert_place:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                if rank == 1:
                    self._inst.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                else:
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                if len(early_backward_batch_index) > 0:
                    insert_place = early_backward_batch_index.pop(0)
                else:
                    insert_place == -1
                
        while len(send_up_pended) > 0:
            self._inst.append(send_up_pended.pop(0))
        
        if num_microbatches_remaining > 0:
            # recv the first steady phase vision forward
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, num_warmup_microbatches, "down"))
        # Run 1F1B in steady state.
        vision_backward_micro_batch_id = 0
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            micro_batch_id = num_warmup_microbatches + i
            self._inst.append(PipelineInstruction(Ops.forward, 0, micro_batch_id, "down"))
            
            if rank == 2 and micro_batch_id == 1:
                backward_batch_id = early_backward_batches.pop(0)
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
            
            self._inst.append(PipelineInstruction(Ops.send_forward_recv_backward, 0, (micro_batch_id, vision_backward_micro_batch_id), "down"))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            if last_iteration:
                self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            else:
                self._inst.append(PipelineInstruction(Ops.send_backward_recv_forward, 0, (vision_backward_micro_batch_id, micro_batch_id + 1), "down"))
            vision_backward_micro_batch_id += 1
        
        # text cool down
        text_backward_micro_batch_id = 0 + early_backward_size        
        # Cool down
        for i in range(num_warmup_microbatches):
            # Backward text cool down
            if text_backward_micro_batch_id < num_microbatches:
                self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
                text_backward_micro_batch_id += 1
            # Backward vision cool down
            self._inst.append(PipelineInstruction(Ops.recv_backward, 0, vision_backward_micro_batch_id, "down"))
            if len(send_up_backward_pend) > 0:
                self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            vision_backward_micro_batch_id += 1
        
        # remaining text cool down
        while len(send_up_backward_pend) > 0:
            self._inst.append(send_up_backward_pend.pop(0))
        while text_backward_micro_batch_id < num_microbatches:
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
            text_backward_micro_batch_id += 1
            
        return self._inst
            
    def format_schedule(self):
        for sche in self._inst:
            vision_or_text = "vision" if sche.model_chunk_id == 0 else "text"
            if hasattr(sche, "recv_micro_batch_id"):
                recv_micro_batch_id = sche.recv_micro_batch_id
                print(f"{vision_or_text}, {sche.op_type}, send micro_batch_id={sche.micro_batch_id}, recv micro_batch_id={recv_micro_batch_id}")
            else:    
                print(f"{vision_or_text}, {sche.op_type}, micro_batch_id={sche.micro_batch_id}")


if __name__ == "__main__":
    sches = Schedules3()
    sches.generate(rank=1, pipeline_parallel_world_size=4, num_microbatches=4, dual_model=True, modal_ratio=2)
    sches.format_schedule()
