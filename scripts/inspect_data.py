import h5py

path = "/home/sungho/bc_ws/robomimic/datasets/lift/ph/low_dim_v15.hdf5"

with h5py.File(path, "r") as f:
    demos = list(f["data"].keys())
    print("데모 개수:", len(demos))

    demo = f["data/demo_0"]
    print("\ndemo_0 행동 모양:", demo["actions"].shape)
    print("demo_0 상태 항목들:")
    for key in demo["obs"].keys():
        print("  ", key, demo["obs"][key].shape)
