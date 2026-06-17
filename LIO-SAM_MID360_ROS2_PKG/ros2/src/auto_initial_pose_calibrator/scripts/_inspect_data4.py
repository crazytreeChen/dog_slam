"""临时脚本: 查看 data_4 的组成"""
import numpy as np, os

base = os.path.dirname(os.path.abspath(__file__))
npz = os.path.join(base, '..', 'scan_viz', 'debug_match_data_4.npz')
d = np.load(npz, allow_pickle=True)

print("Keys:", list(d.keys()))

m = d['map_data']
print(f"\nMap shape: {m.shape}")
print(f"Resolution: {float(d['map_resolution'])}")
print(f"Width: {int(d['map_width'])}, Height: {int(d['map_height'])}")
print(f"Origin: ({float(d['map_origin_x'])}, {float(d['map_origin_y'])})")

unique, counts = np.unique(m, return_counts=True)
total = m.size
print(f"\nMap pixel distribution ({total} total):")
for u, c in zip(unique, counts):
    print(f"  val={int(u):4d}  {c:8d} px  ({c/total*100:5.1f}%)")

tfs = d['frame_tfs']
print(f"\nFrames: {len(tfs)}")
print(f"GT (tf_odom_to_map): {d['tf_odom_to_map']}")
print(f"Frame 0 tf: {tfs[0]}")
print(f"Frame -1 tf: {tfs[-1]}")

# Check frame_ranges keys
range_keys = [k for k in d.keys() if k.startswith('frame_ranges')]
print(f"Range keys: {len(range_keys)}")
ranges0 = d['frame_ranges_0']
print(f"Frame 0 ranges: min={ranges0.min():.2f} max={ranges0.max():.2f} valid(>0)={np.sum(ranges0>0)}/{len(ranges0)}")
