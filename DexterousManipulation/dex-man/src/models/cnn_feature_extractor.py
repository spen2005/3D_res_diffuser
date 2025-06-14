import torch
import torch.nn as nn

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Get input dimensions
        img_shape = observation_space['image'].shape
        if len(img_shape) == 3:  # [H, W, C]
            img_height, img_width = img_shape[0], img_shape[1]
            n_input_channels = img_shape[2]
        else:  # [C, H, W] format
            n_input_channels = img_shape[0]
            img_height, img_width = img_shape[1], img_shape[2]
            
        n_proprio = observation_space['proprioception'].shape[0]

        super().__init__(observation_space, features_dim=256)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the output size of CNN for the actual input dimensions
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, n_input_channels, img_height, img_width)
            ).shape[1]
            
        # Add image feature reduction layer
        self.image_encoder = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.proprio_net = nn.Sequential(
            nn.Linear(n_proprio, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Now combined_net takes reduced image features (128) + proprio features (64)
        self.combined_net = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations['image'].float()
        
        # Permute dimensions from [B, H, W, C] to [B, C, H, W]
        if len(image.shape) == 4 and image.shape[-1] == 3:  # If input is [B, H, W, C]
            image = image.permute(0, 3, 1, 2)

        # Process image through CNN
        image_features = self.cnn(image)
        image_features = self.image_encoder(image_features)

        proprio_features = observations['proprioception'].float()
        proprio_features = self.proprio_net(proprio_features)

        # Combine features
        combined = torch.cat([image_features, proprio_features], dim=1)

        return self.combined_net(combined)