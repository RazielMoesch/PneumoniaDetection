import os
from inference import PneumoniaPredictor
from PIL import Image
import torchvision.transforms.functional as F


###Example of using a Image object
#If doesn't work make sure your in the correct directory
pneumonia_positive_path = "test_images/Pneumonia_True.png"

#open image through path
image = Image.open(pneumonia_positive_path).convert("RGB")
#convert to tensor
tensor = F.to_tensor(image)
#instanciate predictor
predictor = PneumoniaPredictor()
#predict and graph
prediction = predictor.predict(tensor, graph_pred=True)
#print
print(prediction)

