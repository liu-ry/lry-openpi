import dataclasses
from openpi.models import model as _model

import einops
import numpy as np
from openpi import transforms
from PIL import Image  # For resizing


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    if image.shape[:2] != (224, 224):
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((224, 224))  # 插值
        image = np.asarray(pil_img)
    return image

@dataclasses.dataclass(frozen=True)
class ViTaiPI05Inputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        top_image = _parse_image(data["images"]["cam_high"])
        left_image = _parse_image(data["images"]["cam_left_wrist"])
        right_image = _parse_image(data["images"]["cam_right_wrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs



@dataclasses.dataclass(frozen=True)
class ViTaiPI05Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :14])}