# A flexible pipeline implementation for multimodal training.
# Some of the code reuses schedules.py
import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron import get_args
from megatron import print_rank_0, print_rank_all
from megatron import get_timers
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor, get_tensor_shapes


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)
    
    
def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_lengths: list,
    micro_batch_size: int,
    decoder_seq_length: int,
    configs,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    # if config.sequence_parallel:
    #     seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
    #     if model_type == ModelType.encoder_and_decoder:
    #         decoder_seq_length = (
    #             decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
    #         )

    # if model_type == ModelType.encoder_and_decoder:
    #     if parallel_state.is_pipeline_stage_before_split(rank):
    #         tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    #     else:
    #         tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
    #         tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    # else:
 
    if config.v_hidden_size is None:
        # only one hidden-size for all modalities
        for seq_length in seq_lengths:
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_lengths[0], micro_batch_size, config.v_hidden_size))
        tensor_shapes.append((seq_lengths[1], micro_batch_size, config.hidden_size))
    # print(f"tensor_shapes={tensor_shapes}", flush=True)
    # for seq_length, config in zip(seq_lengths, configs):
    #     tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes

def recv_forward(tensor_shapes, config, modal_keys):
    input_tensors = dict()
    for tensor_shape, modal_key in zip(tensor_shapes, modal_keys):
        if tensor_shape is None:
            input_tensors[modal_key] = None
        else:
            input_tensors[modal_key] = p2p_communication.recv_forward(tensor_shape, config)
    return input_tensors


def recv_backward(tensor_shapes, config, modal_keys):
    output_tensor_grads = dict()
    for tensor_shape, modal_key in zip(tensor_shapes, modal_keys):
        if tensor_shape is None:
            output_tensor_grads[modal_key] = None
        else:
            output_tensor_grads[modal_key] = p2p_communication.recv_backward(tensor_shape, config)
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config, modal_keys):
    for (modal_key, tensor_shape) in zip(modal_keys, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensors[modal_key], config)


def send_backward(input_tensor_grads, tensor_shapes, config, modal_keys):
    for (modal_key, tensor_shape) in zip(modal_keys, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grads[modal_key], config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config, modal_keys):
    output_tensor_grads = dict()
    for (modal_key, tensor_shape) in zip(modal_keys, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        if not isinstance(output_tensors, dict):
            # must be loss value of last stage
            out_ = output_tensors
        else:
            out_ = output_tensors[modal_key]
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            out_, tensor_shape, config
        )
        output_tensor_grads[modal_key] = output_tensor_grad
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config, modal_keys):
    input_tensors = dict()
    for (modal_key, tensor_shape) in zip(modal_keys, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grads[modal_key], tensor_shape, config
        )
        input_tensors[modal_key] = input_tensor
    return input_tensors


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if isinstance(config, list):
        config = config[0]
    # print(f"forward_step start", flush=True)
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()
    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True
    
    # print_rank_all(f"call forward step", False)
    # 从model中提取出特有的set_input_tensor方法
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    # 最后的stage需要计算loss
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        # decoder: 同时接收encoder的hidden_state和decoder的input
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    # assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    # assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    # assert isinstance(grad_output, (torch.Tensor, type(None))), (
    #     "grad_output == '%s'." % type(grad_output).__name__
    # )
    if isinstance(output, list):
        for out_, grad_ in zip(output, grad_output):
            Variable._execution_engine.run_backward(
                tensors=(out_,),
                grad_tensors=(grad_,),
                keep_graph=False,
                create_graph=False,
                inputs=tuple(),
                allow_unreachable=True,
                accumulate_grad=True,
            )
    else:
        # Handle scalar output
        if grad_output is None:
            assert output.numel() == 1, "implicit grad requires scalar output."
            grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

        # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
        Variable._execution_engine.run_backward(
            tensors=(output,),
            grad_tensors=(grad_output,),
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, modal_keys):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    
    # FIXME just use config0
    if isinstance(config, list):
        config = config[0]
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()
    # print_rank_all(f"call backward step", False)
    # Retain the grad on the input_tensor.
    for key in modal_keys:
        if input_tensor[key] is not None:
            input_tensor[key].retain_grad()
    # if isinstance(output_tensor, dict):
    #     for key in output_tensor.keys():
    #         if output_tensor_grad[key] is None and config.grad_scale_func is not None:
    #             output_tensor[key] = config.grad_scale_func(output_tensor[key])
    # else:
    #     if output_tensor_grad is None and config.grad_scale_func is not None:
    #         output_tensor = config.grad_scale_func(output_tensor)
    outputs = [output_tensor[key] for key in modal_keys] if isinstance(output_tensor, dict) else output_tensor
    output_grads = [output_tensor_grad[key] for key in modal_keys] if isinstance(output_tensor_grad, dict) else output_tensor_grad
    if not all([False if grad is None else True for grad in output_grads]):
        output_grads = None
    # print_rank_all(f"Before call backward, get outputs={outputs}, output_grads={output_grads}", False)
    
    if config.deallocate_pipeline_outputs:
        custom_backward(outputs, output_grads)
    else:
        torch.autograd.backward(outputs, grad_tensors=output_grads)
    # Collect the grad of the input_tensor.
    input_tensor_grad = dict()
    for key in modal_keys:
        if input_tensor[key] is None:
            input_tensor_grad[key] = None
        else:
            input_tensor_grad[key] = input_tensor[key].grad
    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    # if (
    #     parallel_state.get_pipeline_model_parallel_world_size() > 1
    #     and parallel_state.is_pipeline_stage_after_split()
    #     and model_type == ModelType.encoder_and_decoder
    # ):
    #     if output_tensor_grad[1] is not None:
    #         input_tensor_grad[-1].add_(output_tensor_grad[1])

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def uniform_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: list, # modals may have different seq len.
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    # FIXME currently use the first model's config for trivial functions.
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if configs[0].timers is not None:
        configs[0].timers('forward-backward', log_level=1).start(barrier=configs[0].barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = configs[0].no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()
    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    # print(f"rank={torch.distributed.get_rank()}, total microbatches={num_microbatches}, " + \
    #                 f"num warmup batches={num_warmup_microbatches}", flush=True)

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if configs[0].num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    # self_micro_batch_size = args.xmicro_batch_size \
    #     if parallel_state.is_extra_branch_rank() else args.micro_batch_size
    self_micro_batch_size = args.micro_batch_size
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_lengths=seq_length,
        micro_batch_size=self_micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        configs=configs,
    ) 
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_lengths=seq_length,
        micro_batch_size=self_micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        configs=configs,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    modal_keys = ["image", "text"] # TODO should be configurable
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= configs[0].num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config, modal_keys)
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            configs,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )
        send_forward(output_tensor, send_tensor_shapes, config, modal_keys)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            if isinstance(output_tensor, dict):
                for key in modal_keys:
                    deallocate_output_tensor(output_tensor[key], config.deallocate_pipeline_outputs)
            else:
                # print_rank_all(f"call deallocate_output_tensor, output_tensor={output_tensor}, {isinstance(output_tensors, dict)}", False)
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        # print_rank_all(f"call recv forward, expect shape={recv_tensor_shapes}", False)
        input_tensor = recv_forward(recv_tensor_shapes, config, modal_keys)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= configs[0].num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            configs,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
        )

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config, modal_keys)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config, modal_keys)

        else:
            # print_rank_all(f"forward output_tensor={output_tensor}", False)
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config, modal_keys
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            if isinstance(output_tensor, dict):
                for key in modal_keys:
                    deallocate_output_tensor(output_tensor[key], config.deallocate_pipeline_outputs)
            else:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            # print_rank_all(f"stable pop out output_tensors: {output_tensor}", False)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if configs[0].grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config, modal_keys
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config, modal_keys)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config, modal_keys
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if configs[0].grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            # print_rank_all(f"colldown pop out output_tensors: {output_tensor}", False)

            output_tensor_grad = recv_backward(send_tensor_shapes, config, modal_keys)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config, modal_keys
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config, modal_keys)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            for config in config:
                if config.grad_sync_func is not None:
                    config.grad_sync_func(model.parameters())

    if config.timers is not None:
        config.timers('forward-backward').stop()
        
    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])

    return forward_data_store
