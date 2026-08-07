import os
import scipy.io
import h5py

FILES = [
    os.path.expanduser("~/project/EDCNO/data/raw/burgers_data_R10.mat"),
    os.path.expanduser("~/project/EDCNO/data/raw/piececonst_r421_N1024_smooth1.mat"),
    os.path.expanduser("~/project/EDCNO/data/raw/piececonst_r421_N1024_smooth2.mat"),
]

def inspect_with_scipy(path):
    print(f"\n=== Inspecting with scipy: {path} ===")
    data = scipy.io.loadmat(path)
    for k, v in data.items():
        if k.startswith("__"):
            continue
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"{k}: shape={shape}, dtype={dtype}")

def inspect_with_h5py(path):
    print(f"\n=== Inspecting with h5py: {path} ===")
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(visitor)

for path in FILES:
    try:
        inspect_with_scipy(path)
    except Exception as e:
        print(f"[scipy failed] {path}: {e}")
        try:
            inspect_with_h5py(path)
        except Exception as e2:
            print(f"[h5py failed] {path}: {e2}")