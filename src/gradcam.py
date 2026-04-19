import torch
import torch.nn.functional as F
import numpy as np


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, rr_tensor, target_class=None):
        self.model.eval()

        output = self.model(input_tensor, rr_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self._gradients.mean(dim=2)
        cam = (weights.unsqueeze(2) * self._activations).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0), size=input_tensor.shape[-1],
                            mode='linear', align_corners=False)
        cam = cam.squeeze()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()