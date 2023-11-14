import os
import itertools

def run_grid_search():
    base_cmd = "bash run.sh search GPT-1.3B {} {} {} {} {} nico2 {}"
    tp = [1, 2, 4, 8]
    dp = [1, 2, 4, 8]
    pp = [1, 2, 4, 8]
    global_batch_size = [64, 128]
    micro_batch_size = [4, 8, 16, 32]
    num_layers = [1,2,3,4]
    total_gpu_num = 8

    paral_space = itertools.product(*[tp, dp, pp, global_batch_size, micro_batch_size])
    cout = 0
    for tup in paral_space:
        t, d, p, gbs, mbs = tup
        total = t * d * p
        if gbs % (mbs * d) != 0:
            continue
        if t * d * p == total_gpu_num and not (t == 1 and d == 1 and p == 8) :
            cout += 1
            run_single(t, d, p, gbs, mbs, recompute_granularity="selective")
            for num_layer in num_layers:
                if num_layer > 24//p:
                    break
                run_single(t, d, p, gbs, mbs, recompute_granularity="full", method="uniform", recompute_layers=num_layer)
                run_single(t, d, p, gbs, mbs, recompute_granularity="full", method="block", recompute_layers=num_layer)

def run_group(t, d, p, gbs):
    total = t * d * p
    mbs_list = [4, 8, 16, 32]
    num_layers = [1,2,3,4]
    for mbs in mbs_list:
        if gbs % (mbs * d) != 0:
            continue
        run_single(t, d, p, gbs, mbs, recompute_granularity="selective")
        run_single(t, d, p, gbs, mbs, recompute_granularity="none")
        for num_layer in num_layers:
            if num_layer > 24//p:
                break
            run_single(t, d, p, gbs, mbs, recompute_granularity="full", method="uniform", recompute_layers=num_layer)
            run_single(t, d, p, gbs, mbs, recompute_granularity="full", method="block", recompute_layers=num_layer)

def run_single(tp, dp, pp, gbs, mbs, recompute_granularity = "selective", method = "none", recompute_layers = 0):
    total = tp * dp * pp
    assert gbs % (mbs * dp) == 0, "global batch size should be divisible by micro batch size * data parallelism."
    if recompute_granularity == "selective":
        experiment_method = f"{recompute_granularity}"
    elif recompute_granularity == "full":
        experiment_method = f"{recompute_granularity}_{method}_{str(recompute_layers)}" 
    elif recompute_granularity == "none":
        experiment_method = f"none"

    cmd = f"bash run_silent.sh {experiment_name} {model_name} {tp} {dp} {pp} {gbs} {mbs} {node_name} {total} {recompute_granularity} {method} {str(recompute_layers)}"
    os.system(cmd)
    file_name = f"{model_name}_t{tp}_d{dp}_p{pp}_gbs{gbs}_mbs{mbs}_WAY{experiment_method}.log"
    analysis_single(file_name, experiment_name)


def analysis_single(file_name, experiment_name):
    log_path = f"/home/chen-yy20/Megatron-LM/logs/{experiment_name}/{file_name}"
    import numpy as np
    import re
    with open(log_path, "r") as log:
        lines = log.readlines()
        lines.reverse()
        iter_times = []
        time_cost = 0
        memory_costs = []
        for line in lines:
            if "going to /tmp instead" in line:
                print("NICO ERROR HAPPENED IN {}".format(log_path))
                exit(0)
            if "CUDA out of memory" in line:
                time_cost = "OOM"
                break
            iter_time = re.findall(r"iteration \(ms\): (.+?) \|", line)
            memory_cost = re.findall(r"max allocated: (.+?) \|", line)
            if len(memory_cost) == 0 and len(iter_time) == 0:
                continue
            elif len(memory_cost) != 0:
                memory_costs.append(float(memory_cost[0]))
                print(memory_cost)
            elif len(iter_time) != 0:
                print(iter_time)
                iter_times.append(float(iter_time[0]))
        if time_cost != "OOM":
            time_cost = np.mean(iter_times) / 1000.0
        if len(memory_costs) != 0:
            memory_allocated = np.max(memory_costs)
        else :
            memory_allocated = 0
    t, d, p, gbs, mbs, method = re.findall(r"_t(.+)_d(.+)_p(.+)_gbs(.+)_mbs(.+)_WAY(.+).log", file_name)[0]
    if time_cost == 'OOM':
        throughput_info = f"tp {t}, dp {d}, pp {p}, gbs {gbs}, mbs {mbs}, {method}, CUDA out of memory.\n"
    else:
        thpt = round(float(gbs) / float(time_cost), 2)
        throughput_info = f"{model_name} tp {t}, dp {d}, pp {p}, gbs {gbs}, mbs {mbs}, {method}, {thpt} sample/sec, {memory_allocated} MB.\n"
    print(throughput_info)
    with open(f"/home/chen-yy20/Megatron-LM/logs/{experiment_name}/record.txt", "a") as f:
        f.write(throughput_info)

model_name = "GPT-1.3B"
experiment_name = "throughput_test"
node_name = "nico1"

if __name__ == "__main__":
    # run_grid_search()
    # analysis("search")
    # run_single(2, 2, 2, 64, 8)
    # analysis_single("GPT-760M_t1_d2_p4_gbs128_mbs8.log")
    # analysis_single("GPT-760M_t2_d1_p4_gbs64_mbs4.log")
    # run_single(2, 1, 4, 64, 4, recompute_granularity="full", method="block", recompute_layers=3)
    # run_group(1,4,2,128)
    # run_group(1,2,4,64)
    # run_group(4,1,2,128)
    # run_group(2,4,1,128)
    # run_group(1, 4, 2, 256)
    # run_single(1, 4, 2, 768, 6, "none")
    # run_single(1, 4, 2, 768, 6, "full", "uniform", 1)
    # run_single(1, 4, 2, 768, 6, "full", "uniform", 2)
    # run_single(1, 4, 2, 768, 6, "full", "block", 4)
    run_single(1, 4, 2, 768, 6, "full", "block", 8)
    # run_single(1, 4, 2, 128, 2, "full", "uniform", 1)
    # run_single(1, 4, 2, 128, 4, "full", "uniform", 12)
    # run_single(1, 4, 2, 128, 8, "full", "uniform", 1)
    # run_single(2, 2, 2, 64, 1)
    # run_single(2, 1, 4, 64, 16)
    # run_single(2, 1, 4, 64, 32)
    # run_single(1, 4, 2, 128, 1)