import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms.functional as F
import os
from PIL import Image
from model import Pneumonia_Model
import matplotlib.pyplot as plt


class PneumoniaPredictor():
  def __init__(self, model_path=None):
    self.model = Pneumonia_Model(model_path=model_path).model
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  

  def predict(self, image, return_pred=True, graph_pred=False, return_class_scores=False):
    '''
    Using a simple classification model predict whether a patient has pneumonia based on an xray of the lungs

    Parameters:
        image: The image to predict on | Expected: PIL.Image, torch.Tensor, Path
        return_pred: Decide whether you want to return the prediction | Expected: Boolean | Default:True
        graph_pred: Decide whether you want to graph the prediction along with the image | Expected: Boolean | Default:False
        return_class_scores: Decide whether you want to return the scores for each class | Expected: Boolean | Default:False

        *Note if both return_pred and return_class_scores are 'True' only return_pred will be active*
    
    Return:
      Default: return_pred=True | Returns Prediction

    '''
    if isinstance(image, Image.Image):
      image = F.to_tensor(image)
    elif isinstance(image, torch.Tensor):
      image = image
    elif isinstance(image, str):
      if os.path.exists(image):
        image = Image.open(image).convert("RGB")
        image = F.to_tensor(image)
      else:
        raise FileNotFoundError("The path given was not found.")
    else:
      raise ValueError("Input must be a PIL Image, Tensor, or Path")
    plt_image = image
    w = image.size()[2]
    h = image.size()[1]
    #print(w, h)
    
    
    self.model.to(self.device)
    self.model.eval()
    
    with torch.inference_mode():
      image = transforms.Grayscale()(image)
      image = transforms.Resize((244,244))(image)
      cls_scores = self.model(image.unsqueeze(0).to(self.device))
    pred = torch.argmax(cls_scores)
    #print(pred)

    image = plt_image.permute(1,2,0).cpu()
    image = image.numpy()
    if graph_pred:
      fig, ax = plt.subplots()
      ax.imshow(image)
      ax.text(w/2, h*.10, f"Guess: {pred==1}", size=20, color='r', horizontalalignment='center', 
            verticalalignment='center')
      plt.show()
    
    if return_pred:
      return (pred == 1).item()
    
    if return_class_scores:
      return cls_scores






