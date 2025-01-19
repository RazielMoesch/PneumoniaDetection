import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os





class BasicClassificationModel(nn.Module):
  def __init__(self, num_classes, input_shape=1, hidden_units=16, image_size=(244,244)):
    super().__init__()

    self.image_size = image_size
    self.input_shape=input_shape

    self.convlayer1 = nn.Sequential(
                                    nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    )
    
    self.convlayer2 = nn.Sequential(
                                    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    )

    self.convlayer3 = nn.Sequential(
                                    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16),
                                    nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    )

    self.flatten = nn.Flatten()


    dummy = torch.zeros(1, input_shape, *image_size)
    with torch.no_grad():
      dummy = self.convlayer1(dummy)
      dummy = self.convlayer2(dummy)
      #dummy = self.convlayer3(dummy)
      lin_features = dummy.numel()
      

    
    self.LinearLayer1 = nn.Linear(in_features=lin_features, out_features=num_classes)
  

  def forward(self, x):
    # if x[0][0] != self.input_shape:
    #   raise ValueError(f"Expected an image with {self.input_shape} dimensions not {x[0][0]}. By default the model expects images to be in GrayScale")
    
    # x = F.interpolate(x, size=self.image_size, mode="bilinear")
    x = self.convlayer1(x)
    x = self.convlayer2(x)
    x = self.flatten(x)
    x = self.LinearLayer1(x)

    return x
  
  

class Pneumonia_Model():
  def __init__(self, model_path="models/pneumonia_dict_0.pth", device=None):
    if not device:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device = device
    if not model_path:
      self.model_path = "models/pneumonia_dict_0.pth"
    else:
      self.model_path = model_path
    self.model = BasicClassificationModel(2)

    self.load_model()

    
 

  def load_model(self):
    print(self.model_path)
    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))