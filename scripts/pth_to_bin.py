import os
import argparse

import torch
import numpy as np

def main(model_path: str, output_dir: str):
    # 1) Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # 2) Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # 3) Iterate parameters
    for name, tensor in state_dict.items():
        # ensure a CPU FloatTensor
        arr = tensor.cpu().numpy().astype(np.float32)

        # sanitize filename
        fname = name.replace(".", "_") + ".bin"
        out_path = os.path.join(output_dir, fname)

        # write raw binary float32 LE
        arr.tofile(out_path)
        print(f"Wrote {name:30s} → {out_path}  (shape={arr.shape})")

    print(f"\nAll done! {len(state_dict)} files written to {output_dir!r}.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Dump each parameter in a .pth to individual .bin files"
    )
    p.add_argument(
        "--model-path", "-m", required=True,
        help="Path to your model.pth (or checkpoint) file"
    )
    p.add_argument(
        "--output-dir", "-o", default="bins",
        help="Directory to write .bin files into"
    )
    args = p.parse_args()
    main(args.model_path, args.output_dir)
