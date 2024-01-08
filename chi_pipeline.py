"""Adapted from https://github.com/Shigangli/Chimera"""
from enum import Enum, auto

class PPOps(Enum):
    forward = auto()
    backward = auto()
    send_forward = auto()
    recv_forward = auto()
    send_backward = auto()
    recv_backward = auto()
    synchronize = auto()
    
class PipelineInstruct():
    def __init__(self, op_type, model_chunk_id=None, up_or_down=None) -> None:
        self.op_type = op_type
        self.model_chunk_id = model_chunk_id
        self.up_or_down = up_or_down

class AutoGeneratePipelineRank:

    def __init__(self, stage_numbers, divisors, micro_batch_numbers):
        self.module_to_stage_map = [i for i in range(stage_numbers)]
        self.stage_numbers = stage_numbers
        assert divisors % 2 == 0, "pipeline num must be an even number"
        self.pipeline_numbers = 1 if divisors is None else divisors//2 # 总的模型划分数除以二，因为是双向pipeline, 一般就设置为1就好了
        self.micro_batch_numbers = micro_batch_numbers
        self.push_pipeline_numbers = {
            "up": 0,
            "down": 0
        }
        self.push_micro_batch = 0 # 已经分配的micro batch数
        self.total_cnt = 0

    def generate_pipeline(self):
        self.up_pipline_list = []
        self.down_pipeline_list = []
        for i in range(self.pipeline_numbers): # 一共也就执行一次
            # generate up pipeline
            micro_num = self.stage_numbers//(2*self.pipeline_numbers) # 总阶段数除以pipeline数的两倍，每一轮分配的micro batch数

            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num
            print(f"micro_num {micro_num}")
            self.up_pipline_list.append(MyPipeLine(i,
                                                   micro_num, self.stage_numbers,
                                                   self.pipeline_numbers, self.module_to_stage_map, True))

            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num

            # generate down pipeline
            self.down_pipeline_list.append(MyPipeLine(i,
                                                    micro_num, self.stage_numbers,
                                                    self.pipeline_numbers, self.module_to_stage_map, False))

    def get_schedule(self, is_iteration=False):
        schedule = []
        schedule_up_down = []
        pipelines = self.up_pipline_list + self.down_pipeline_list

        for i in range(self.stage_numbers):
            schedule.append(list())
            schedule_up_down.append(list())
        has_next_flag = True
        has_next_sync = 0
        steps = 0
        sync_list = [[] for i in range(self.stage_numbers)]
        while(has_next_flag or has_next_sync != 0): # 流水线还有下一步或者还有同步信息
            next_flag = False
            sub_schedule = list("" for _ in range(self.stage_numbers))
            for index, pipeline in enumerate(pipelines):
                if pipeline.has_next_pass():
                    next_data, is_pop, step_direction, up_or_down, is_sync = pipeline.next_pass()
                    # print(f"next data {next_data}")
                    for k in next_data.keys():
                        schedule[next_data[k] %
                                 self.stage_numbers].append(str(k)) # 更新schedule
                        up_or_down_str = str(index)
                        up_or_down_str += "@down@" if up_or_down else "@up@"
                        schedule_up_down[next_data[k] % # 更新schedule_up_down
                                         self.stage_numbers].append(f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}")
                        if step_direction.get(pipeline.micro_batch_ids[-1], 1) != 1:
                            direction = "down"
                            if pipeline.up_or_down:
                                direction = "up"

                        sub_schedule[next_data[k] % # 更新sub_schedule
                                     self.stage_numbers] = f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}"
                    if is_sync and next_data.get(pipeline.micro_batch_ids[-1]) is not None:
                        has_next_sync += 1
                        sync_list[next_data[pipeline.micro_batch_ids[-1]] %
                                  self.stage_numbers].append(f"{up_or_down_str}s")
                    # FIXME: add not_finished
                    not_finished = (self.micro_batch_numbers - self.push_micro_batch) > 0 if self.stage_numbers == 2 else False
                    # not_finished = False
                    if is_pop and (pipeline.has_next_pass() or not_finished): # 已经有微批次走完了一轮流水线，且还有下一轮
                        micro_num = self.stage_numbers//(2 *
                                                         self.pipeline_numbers)
                        if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                            micro_num = self.micro_batch_numbers-self.push_micro_batch

                        self.push_micro_batch += micro_num
                        if micro_num != 0:
                            if pipeline.up_or_down:
                                direction = "up"
                            else:
                                direction = "down"
                            # 生成新的流水线
                            pipelines.append(MyPipeLine(self.pipeline_numbers+self.push_pipeline_numbers[direction],
                                                        micro_num, self.stage_numbers,
                                                        self.pipeline_numbers, self.module_to_stage_map, pipeline.up_or_down))
                            self.push_pipeline_numbers[direction] += 1

                    next_flag = True
            print(f"schedule {schedule} | schedule_up_down {schedule_up_down} | sub {sub_schedule}")
            self.total_cnt += 1
            temp_mb_ids = []
            for idex, mb_list in enumerate(schedule):
                if len(mb_list) == self.total_cnt:
                    temp_mb_ids.append(mb_list[-1])
                else:
                    if idex == 0:
                        print(f"{self.total_cnt}: {len(mb_list)}")
                    temp_mb_ids.append("")

            # 如果有同步信息且当前步骤的 sub_schedule 中有空位，将同步信息填充到空位中。
            for index, s in enumerate(sub_schedule):
                if s == "" and len(sync_list[index]) > 0:
                    sub_schedule[index] = sync_list[index].pop(0)
                    has_next_sync -= 1
            # 将当前步骤的 sub_schedule 更新到整体的调度表中。
            for i in range(self.stage_numbers):
                if len(schedule[i]) <= steps:
                    schedule[i].append(sub_schedule[i])
                    schedule_up_down[i].append(sub_schedule[i])

            steps += 1
            has_next_flag = next_flag
            if is_iteration and has_next_flag:
                # input("pause")
                yield temp_mb_ids, sub_schedule # 返回当前步骤的调度表，以后调用 next() 时，会从这里继续执行。
                
    def get_exec_schedule(self, self_stage_id, warm_up_batches):
        """Computation schedules of bidirectional pipeline cannot be launched in order,
        because the opposite send ops will block the schedule. We rearrange the order
        to make the op list can execute one-by-one.
        """
        
        schedule_pipeline = self.get_schedule(True)
        pipeline_schedule = []
        for sub_schedule in schedule_pipeline:
            pipeline_schedule.append(sub_schedule)
        local_device_ops = []
        
        def is_cross(current_sched, next_sched):
            current_sched
            
        for batch_id, sub_schedule in enumerate(pipeline_schedule):
            if sub_schedule[self_stage_id] != '':
                index, up_down, forward_backward = sub_schedule[self_stage_id].split("@")
                index = int(index) # model index located in self device
                
                # if batch_id < warm_up_batches:
                #     # warmup phase
                #     assert forward_backward == 'f'
                #     local_device_ops.append(PipelineInstruct(PPOps.forward, index, up_down))
                #     local_device_ops.append(PipelineInstruct(PPOps.send_forward, index, up_down))
        
                    # up pipeline delay send
                if forward_backward == 'f':
                    local_device_ops.append(PipelineInstruct(PPOps.recv_forward, index, up_down))
                    local_device_ops.append(PipelineInstruct(PPOps.forward, index, up_down))
                    local_device_ops.append(PipelineInstruct(PPOps.send_forward, index, up_down))
                if forward_backward == 'b':
                    local_device_ops.append(PipelineInstruct(PPOps.recv_backward, index, up_down))
                    local_device_ops.append(PipelineInstruct(PPOps.backward, index, up_down))
                    local_device_ops.append(PipelineInstruct(PPOps.send_backward, index, up_down))
                if forward_backward == 's':
                    local_device_ops.append(PipelineInstruct(PPOps.synchronize, index, up_down))
                

class MyPipeLine:
    def __init__(self, pipeline_id, micro_batch_numbers,
                 stage_numbers, pipeline_numbers, module_to_stage_map, up_or_down):

        self.pipeline_id = pipeline_id
        self.micro_batch_numbers = micro_batch_numbers # 应该只是对应一轮
        self.stage_to_rank_map = None
        self.pipeline_numbers = pipeline_numbers
        self.stage_numbers = stage_numbers
        self.module_to_stage_map = module_to_stage_map # 模块到流水线阶段的映射 range(stage_numbers)
        self.up_or_down = up_or_down
        self.devices = None

        self.steps = -1
        self.step_direction = dict()
        self.micro_batch_ids = list()
        self.micro_batch_device = dict()
        # FIXME:micro_batch_id = ((self.pipeline_id//2)*self.stage_numbers) # 0
        # micro_batch_id += (0 if self.up_or_down else self.stage_numbers//2) # 0 2
        micro_batch_id = (0 if self.up_or_down else self.stage_numbers//2) # 0 2
        print("初始化！====================================")
        for x in range(self.micro_batch_numbers): # 每条流水线分别处理一半的micro batch
            # FIXME:self.micro_batch_ids.append(
            #     x+(self.pipeline_id % self.pipeline_numbers)*(self.stage_numbers//self.pipeline_numbers //2)+micro_batch_id)
            self.micro_batch_ids.append(
                x+(self.pipeline_id)*(self.stage_numbers//self.pipeline_numbers)+micro_batch_id)
        print(f"id {self.pipeline_id} | p_num {self.pipeline_numbers} | mb_id {micro_batch_id} | micro_batch_ids {self.micro_batch_ids}")
        start_stage_device = (self.pipeline_id % self.pipeline_numbers) * \
            (self.stage_numbers // self.pipeline_numbers)
        self.devices = [x for x in self.module_to_stage_map[start_stage_device:] +
                        self.module_to_stage_map[:start_stage_device]]
        if self.up_or_down is True:
            # down pipeline
            self.stage_to_rank_map = {
                str(index): [device] for index, device in enumerate(self.devices)}
        else:
            # up pipeline
            self.stage_to_rank_map = {
                str(self.stage_numbers-1-index): [device] for index, device in enumerate(self.devices)}

    def next_pass(self):# 模拟下一个处理步骤，返回当前处理的micro batch id，是否需要pop，当前步骤的方向，是否需要同步
        if self.steps <= (self.micro_batch_numbers-1) * 2:
            self.steps += 1

        over_back_micro_batch = []
        for micro_batch in self.micro_batch_device.keys(): # 遍历micro_batch_device中的所有micro batch
            step = 1 if self.up_or_down else -1 # 根据流水线方向确定步进方向
            if self.step_direction[micro_batch] == 1 and abs(self.micro_batch_device[micro_batch] - (self.stage_to_rank_map["0"][0] + 2*self.stage_numbers)) >= self.stage_numbers-1:
                self.step_direction[micro_batch] = -1 # 如果微批次当前的步进方向是正向，并且已经移动到了该阶段的起始阶段之后，将步进方向改为负向。
            elif self.step_direction[micro_batch] == -1 and self.micro_batch_device[micro_batch] == self.stage_to_rank_map["0"][0] + 2*self.stage_numbers:
                over_back_micro_batch.append(micro_batch) # 如果微批次当前的步进方向是负向，并且已经回到了该阶段的起始阶段，将该微批次添加到 over_back_micro_batch 列表中。
            else:
                self.micro_batch_device[micro_batch] += step * \
                    self.step_direction[micro_batch] # 如果微批次还可以继续前进或后退，更新微批次的位置。
        # print(f"p_id {self.pipeline_id} | mb_device {self.micro_batch_device}")
        pop_one = False
        for micro_batch in over_back_micro_batch:
            self.micro_batch_device.pop(micro_batch) # 将 over_back_micro_batch 列表中的微批次从 micro_batch_device 中删除，表示已经完成了一轮流水线的前进或后退。
            print(f"{micro_batch} is popped.")
            pop_one = True

        if self.steps % 2 == 0: # 偶数步往micro_batch_device中添加新的微批次
            self.micro_batch_device[self.micro_batch_ids[self.steps //
                                                         2]] = self.stage_to_rank_map["0"][0] + 2*self.stage_numbers
            
            self.step_direction[self.micro_batch_ids[self.steps // 2]] = 1 # 微批次的步进方向为前向传播
        is_sync = True if self.step_direction.get(
            self.micro_batch_ids[-1]) == -1 else False  # 如果最后一个微批次开始反向传播，则可以开始同步
        return self.micro_batch_device, pop_one, self.step_direction, self.up_or_down, is_sync

    def has_next_pass(self):
        # print(f"has_next_pass {self.micro_batch_numbers} {self.micro_batch_device}")
        if self.micro_batch_numbers > 0 and (self.steps == -1 or self.micro_batch_device):
            return True
        return False


# ##########################测试用
def forward(index, up_or_down):
    print(f"forward {index}:{up_or_down}")


def backward(index, up_or_down):
    print(f"backward {index}:{up_or_down}")


if __name__ == "__main__":
    stage_num = 4
    pipeline_num = 2
    micro_num = 20
    print(f"stage:{stage_num}  pipeline_num:{pipeline_num} micro_num:{micro_num}")
    pipeline = AutoGeneratePipelineRank(stage_num, pipeline_num, micro_num)
    # generate intstruction
    pipeline.generate_pipeline()
    schedule_pipeline = pipeline.get_schedule(True)
    pipeline_schedule = []
    mbs_ids = []
    for mb_ids, sub_schedule in schedule_pipeline:
        mbs_ids.append(mb_ids)
        pipeline_schedule.append(sub_schedule)

    cout = 0
    stage_model_id = 0
    # print(f"schedule: {pipeline_schedule}")
    # execute instructions
    for mb_ids, sub_schedule in zip(mbs_ids, pipeline_schedule):
        print(f"step{cout}: {sub_schedule} | {mb_ids}")
        cout += 1
        # if sub_schedule[stage_model_id] != '':
        #     index, up_down, forward_backward = sub_schedule[stage_model_id].split(
        #         "@")
        #     index = int(index)  
        #     if forward_backward == 'f':
        #         forward(index, up_down)
        #     elif forward_backward == 'b':
        #         backward(index, up_down)
        #     elif forward_backward == 's':
        #         print(f"sync {index}, {up_down}")
        print("=====")