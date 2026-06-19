import shutil
import sys
from datetime import datetime
from pathlib import Path

realsense_path, ioi_path = Path(sys.argv[1]), Path(sys.argv[2])

# NOTE: Temporary measure for dealing with the fact that the Maira Box does not sync its clock to the internet.
offset = 135


def split_format(n):
    return datetime.strptime(n, "%Y%m%d%H%M%S%f")


def split_format_us(n):
    return datetime.strptime(n, "%m%d%Y%H%M%S%f")


def time_diff(t1, t2):
    return abs((t1 - t2).total_seconds() + offset)


ioi_frames = [(p, split_format_us(p.stem)) for p in ioi_path.iterdir()]
realsense_frames = [(p, split_format(p.stem)) for p in realsense_path.iterdir()]

keep_frames = []
for p_ioi, t_ioi in ioi_frames:
    min_rs = min(realsense_frames, key=lambda rs: time_diff(t_ioi, rs[1]))
    p_rs, t_rs = min_rs
    print(f"min frame for {p_ioi} --> {p_rs}: diff={time_diff(t_ioi, t_rs)}")
    keep_frames.append((p_ioi, p_rs))

save_dir = (
    realsense_path.parent / f"{ioi_path.stem}_to_{realsense_path.stem}_keep"
)
save_dir.mkdir(parents=True, exist_ok=True)
for p_ioi, p_rs in keep_frames:
    shutil.copy(p_rs, save_dir / f"{p_ioi.stem}_{p_rs.stem}{p_rs.suffix}")
