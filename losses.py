import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import Normalize
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualLoss(nn.Module):
    """
    A custom PyTorch module for calculating perceptual loss using VGG19 features.

    Attributes:
    - model: A pre-trained VGG19 model.
    - feature_extractor: A feature extractor from the VGG19 model.
    - normalize: A normalization layer for input tensors.

    Methods:
    - __init__(self, feature_extraction_node="features.35", feature_normalize_mean=(0.485, 0.456, 0.406),
                 feature_normalize_std=(0.229, 0.224, 0.225)): Initializes the PerceptualLoss module.
    - forward(self, sr_tensor, gt_tensor): Computes the perceptual loss between the super-resolved (sr_tensor) and ground truth (gt_tensor) tensors.
    """

    def __init__(self, feature_extraction_node="features.35", feature_normalize_mean=(0.485, 0.456, 0.406),
                 feature_normalize_std=(0.229, 0.224, 0.225)):
        super(PerceptualLoss, self).__init__()
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(self.model, list(feature_extraction_node))
        self.normalize = Normalize(feature_normalize_mean, feature_normalize_std)

        # disable gradients
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor, gt_tensor):
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        # input normalization
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Get the output of the feature extraction layer
        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Compute feature loss
        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F.mse_loss(sr_feature[self.feature_extractor_nodes[i]],
                                     gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses
