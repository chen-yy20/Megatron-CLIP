from pty import slave_open
import torch
import time
import copy


_GLOBAL_TIMERS = None

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_timers():
    """Return timers."""
    if _GLOBAL_TIMERS is None:
        print("Timer has not been initialized, init..")
        _set_timers()
    return _GLOBAL_TIMERS

def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()

def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)

def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}
        self.__accu_timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:   
            print(string, flush=True)

    def log_all(self, normalizer=1.0, logger=None):
        """ Call for each iteration to 1) calculate elapse time; 
            2) format elapse time and print it out.
        """
        assert normalizer > 0.0
        string = 'time (ms)'
        for timer_key in self.timers:
            elapsed_time = self.timers[timer_key].elapsed(
                reset=True) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(timer_key, elapsed_time)
            if timer_key not in self.__accu_timers.keys():
                self.__accu_timers[timer_key] = []
            self.__accu_timers[timer_key].append(elapsed_time)
        if logger is not None:
            logger.info(string)
        else:
            # if torch.distributed.is_initialized():
            #     if torch.distributed.get_rank() == (
            #             torch.distributed.get_world_size() - 1):
            #         print(string, flush=True)
            # else:
            print(string, flush=True)

    def log_avg(self, final_avg_batch=10):
        """ Call after all iteration to statistic average elapse time
        """
        string = f"Avg of the last {final_avg_batch} batches, time (ms)"
        for timer_key, cost in self.__accu_timers.items():
            string += ' | {}: {:.2f}'.format(timer_key, sum(cost[-final_avg_batch:]) / final_avg_batch)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)