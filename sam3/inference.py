import os
import matplotlib.pyplot as plt
from torchvision.transforms import v2

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
import torch


class SAM3():
    def __init__(self):
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path)

    def inference(self, image, prompt, confidence_threshold=0.5, debug=False):
        """
        Args:
            image: PIL Image / np.array or torch.Tensor of shape [C,H,W]
            prompt: text prompt
            confidence_threshold: threshold for filtering predictions based on confidence score
            debug: if True, will plot the results
        """

        processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

        if debug:
            plot_results(image, inference_state)
            plt.show()


if __name__ == "__main__":
    device = "cuda:0"
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    image_path = f"{sam3_root}/assets/images/franka_tabletop4.jpg"
    image = Image.open(image_path)
    image_tensor = v2.ToTensor()(image)              # uint8 [C,H,W]
    image_tensor = v2.ToDtype(torch.float32, scale=True)(image_tensor).to(device) # float32 [C,H,W] in [0,1]

    sam3 = SAM3()
    sam3.inference(
        image=image,
        prompt="smallest black cube",
        confidence_threshold=0.5,
        debug=True
    )

