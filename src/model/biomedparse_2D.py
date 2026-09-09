"""2D inference using the v2 mask decoder and existence classifier."""

import torch

from .biomedparse_3D import BiomedParseModel as VolumeModel


class BiomedParseModel(VolumeModel):
    def forward(self, inputs, mode="eval"):
        if mode != "eval":
            raise ValueError("Use the existing biomedparse model/configuration for training.")
        return self.forward_eval(inputs)

    @torch.no_grad()
    def forward_eval(self, inputs):
        image = inputs.get("image")
        text = inputs.get("text")
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("image must be a tensor of shape [B, 1 or 3, H, W].")
        if image.shape[0] == 0 or image.shape[1] not in (1, 3):
            raise ValueError("image must contain a nonempty batch with 1 or 3 channels.")
        if min(image.shape[-2:]) <= 0:
            raise ValueError("Image dimensions must be positive.")
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, (list, tuple)) or len(text) != image.shape[0]:
            raise ValueError("Provide one prompt string per image, separated by [SEP] for multiple targets.")
        if any(not isinstance(prompt, str) or any(not part.strip() for part in prompt.split("[SEP]")) for prompt in text):
            raise ValueError("Text prompts must be nonempty strings.")
        mean, std = self.pixel_mean, self.pixel_std
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
            if torch.all(mean == mean.flatten()[0]) and torch.all(std == std.flatten()[0]):
                mean, std = mean.mean(), std.mean()
        image = (image - mean) / std
        prompt_features = self.sem_seg_head.encode_prompts(text, eval=True)
        outputs = self.sem_seg_head(
            image_features=self.backbone(image), prompt_features=prompt_features,
        )
        num_prompts = prompt_features["num_prompts"].to(image.device)
        outputs["num_prompts"] = num_prompts
        if self.edge_queries > 0:
            outputs["edge_masks"] = outputs["pred_gmasks"][:, -self.edge_queries:].mean(dim=1, keepdim=True)
            outputs["pred_gmasks"] = outputs["pred_gmasks"][:, :-self.edge_queries]
        else:
            outputs["edge_masks"] = None
        if self.convolute_outputs:
            outputs["pred_gmasks"] = self.convolution_procedure(
                image.repeat_interleave(num_prompts, dim=0), outputs["pred_gmasks"],
            )
        else:
            outputs["pred_gmasks"] = outputs["pred_gmasks"].mean(dim=1, keepdim=True)
        return {"predictions": outputs}