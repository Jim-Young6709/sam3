import os
import io
import json
import time
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.model.eval()

    @torch.inference_mode()
    def inference(self, image: torch.Tensor, prompt: str, confidence_threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Args:
            image: torch.Tensor [C,H,W] float32 in [0,1], on self.device preferred
            prompt: text prompt
        Returns:
            dict of numpy arrays: masks (N,H,W), boxes (N,4), scores (N,)
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be torch.Tensor, got {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"image must be [C,H,W], got shape {tuple(image.shape)}")
        if image.device != self.device:
            image = image.to(self.device, non_blocking=True)

        processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)

        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        # Convert outputs to numpy
        masks = state["masks"][:, 0, :, :].detach().cpu().numpy()   # (N,H,W)
        boxes = state["boxes"].detach().cpu().numpy()              # (N,4)
        scores = state["scores"].detach().cpu().numpy()            # (N,)

        return {"masks": masks, "boxes": boxes, "scores": scores}


def _decode_image_from_bytes(meta: Dict[str, Any], payload: bytes, device: torch.device) -> torch.Tensor:
    """
    meta fields:
      - shape: [H,W,3] or [3,H,W]
      - dtype: "uint8" or "float32"
      - layout: "HWC" or "CHW"
      - color: "RGB"
      - normalize: bool (if dtype is uint8 -> convert to float32/255 when True)
    """
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    layout = meta["layout"]
    normalize = bool(meta.get("normalize", True))

    arr = np.frombuffer(payload, dtype=dtype)
    arr = arr.reshape(shape)

    if layout == "HWC":
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
    elif layout != "CHW":
        raise ValueError(f"Unsupported layout: {layout}")

    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr)  # uint8 CHW
        if normalize:
            t = t.to(torch.float32).div_(255.0)
        else:
            t = t.to(torch.float32)
    elif arr.dtype == np.float32:
        t = torch.from_numpy(arr)  # float32 CHW
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")

    # Ensure contiguous and move
    return t.contiguous().to(device, non_blocking=True)


def _encode_outputs_to_npz(outputs: Dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    # masks can be large; np.savez_compressed helps a lot
    np.savez_compressed(buf, **outputs)
    return buf.getvalue()


def main():
    import zmq

    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="tcp://172.16.0.7:5555")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rcv_timeout_ms", type=int, default=0)  # 0 = block forever
    parser.add_argument("--snd_timeout_ms", type=int, default=0)
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)

    if args.rcv_timeout_ms > 0:
        sock.RCVTIMEO = args.rcv_timeout_ms
    if args.snd_timeout_ms > 0:
        sock.SNDTIMEO = args.snd_timeout_ms

    model = SAM3(device=args.device)
    print(f"[sam3_server] bound to {args.bind}, device={args.device}")
    print("--------------------- SAM3 Server is Ready ---------------------")
    
    while True:
        try:
            # Expect: [meta_json_bytes, image_bytes]
            meta_b, img_b = sock.recv_multipart()
            meta = json.loads(meta_b.decode("utf-8"))

            prompt = str(meta["prompt"])
            conf = float(meta.get("confidence_threshold", 0.5))

            t0 = time.time()
            image_t = _decode_image_from_bytes(meta["image"], img_b, device=model.device)
            outputs = model.inference(image=image_t, prompt=prompt, confidence_threshold=conf)
            npz_blob = _encode_outputs_to_npz(outputs)
            dt = time.time() - t0

            resp_meta = {"ok": True, "dt_s": dt}
            sock.send_multipart([json.dumps(resp_meta).encode("utf-8"), npz_blob])

        except Exception as e:
            err = {"ok": False, "error": repr(e)}
            # send an empty payload on error
            sock.send_multipart([json.dumps(err).encode("utf-8"), b""])


if __name__ == "__main__":
    main()