# For logging info in multi-woker scenarios.
log_base_path = "/home/zanzong/workspace/flexpipe/models/open_clip/timelines/"
# clear logs for this run
import shutil,os
from datetime import datetime
for root, dirs, files in os.walk(log_base_path):
    print(files)
    if len(files) != 0:
        now = datetime.now()
        dtime = now.strftime("%m-%d_%H-%M-%S")
        backup_dir = f"{log_base_path}backup_{dtime}"
        if os.path.exists(backup_dir):
            backup_dir = backup_dir + "_0"
        os.mkdir(backup_dir)
        for file in files:
            shutil.move(os.path.join(root, file), backup_dir)
    break
