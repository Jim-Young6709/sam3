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
        Returns:
            inference_state: a dictionary containing the intermediate states and final predictions of the model
                "original_height": original height of the input image (counted by # pixels)
                "original_width": original width of the input image (counted by # pixels)
                "backbone_out": the output feature map from the backbone: 'vision_features', 'vision_pos_enc', 'backbone_fpn', 'sam2_backbone_out', 'language_features', 'language_mask', 'language_embeds'
                "geometric_prompt":
                "masks_logits":
                "masks": predicted masks, (num_objects, 1, H, W)
                "boxes": predicted bounding boxes, [x0, y0, x1, y1] (top-left and bottom-right corners), wrt original image pixel coordinates, x-axis is width and y-axis is height
                "scores": confidence scores for each predicted box
        """

        processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

        if debug:
            plot_results(image, inference_state)
            plt.show()

        # convert relevant output to numpy
        output_dict= {
            "masks": inference_state["masks"][:, 0, :, :].cpu().numpy(), # (num_objects, H, W)
            "boxes": inference_state["boxes"].cpu().numpy(), # (num_objects, 4)
            "scores": inference_state["scores"].cpu().numpy(), # (num_objects,)
        }
        return output_dict


if __name__ == "__main__":
    device = "cuda:0"
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    image_path = f"{sam3_root}/assets/images/franka_tabletop4.jpg"
    image = Image.open(image_path)
    image_tensor = v2.ToTensor()(image)              # uint8 [C,H,W]
    image_tensor = v2.ToDtype(torch.float32, scale=True)(image_tensor).to(device) # float32 [C,H,W] in [0,1]
    image_tensor_downsampled = image_tensor[:, ::16, ::16]  # Downsample the image by a factor of 2, 260x360

    sam3 = SAM3()
    sam3.inference(
        image=image_tensor,
        prompt="black cube",
        confidence_threshold=0.5,
        debug=True
    )

