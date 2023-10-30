import os, sys
import shutil
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import math
from datetime import datetime
import numpy as np


plt.rcParams.update({
    'lines.markersize': 2,
})


def timeline(logs_dir, out_name=None):
    log_paths = list()
    
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            log_paths.append(os.path.join(root, file))
        break # only consider level 1 dir
    if len(log_paths) == 0:
        print("Empty log dir, exit.")
        exit()
    
    extract_name = lambda path: os.path.basename(path).split(".")[0]
    traces = {extract_name(p): [] for p in log_paths}
    for log_path in log_paths:
        log_name = extract_name(log_path)
        with open(log_path, 'r') as log:
            while True:
                line = log.readline()
                if line is None or line == "":
                    break
                items = line.strip().split("|")
                assert len(items) == 2, f"each line should contain cmd and timestamp. {items}"
                cmd = items[0]
                ts = items[1]
                traces[log_name].append((cmd, float(ts)))
        traces[log_name] = sorted(traces[log_name], key=lambda x: x[1])

    # plot
    plt.figure()
    y_ticks = []
    # log files named with `global_rank_id`
    key_sorted = sorted(traces.keys(), key=lambda x: int(x))
    for idx, log in enumerate(key_sorted):
        records = traces[log]
        timestamps = [r[1] for r in records]
        tags = [r[0] for r in records]
        y_idx = [idx for _ in range(len(records))]
        plt.plot(timestamps, y_idx, marker='o', )
        for idx, t in enumerate(timestamps):
            plt.text(t, y_idx[0] - 0.05, tags[idx] + '-', rotation=90, ha='center', va='top', fontdict={'size': 1.5})
        y_ticks.append(log)
        
    plt.yticks(list(range(len(y_ticks))), y_ticks)
    plt.xlabel("Timeline (s)")
    plt.ylabel("Log name")
    plt.ylim(-1, len(y_ticks) + 1)
    
    if out_name is not None:
        plt.savefig(f"{out_name}.pdf", bbox_inches='tight')
    else:
        now = datetime.now()
        dtime = now.strftime("%m-%d_%H-%M-%S")
        plt.savefig(f"{dtime}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', type=str, help='log dir.', default="None")
    parser.add_argument('name', type=str, help='file name of output fig.', default="None")
    args = parser.parse_args()
    log_dir_path = args.path
    name = args.name
    assert log_dir_path is not None, "Log dir is required!"

    timeline(log_dir_path, name)