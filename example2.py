import os
from inference import PneumoniaPredictor
from PIL import Image


###Example of using a Image object
#If doesn't work make sure your in the correct directory
pneumonia_positive_path = "test_images/Pneumonia_True.png"
#open image through path
image = Image.open(pneumonia_positive_path).convert("RGB")
#instanciate predictor
predictor = PneumoniaPredictor()
#predict and graph
prediction = predictor.predict(image, graph_pred=True)
#print
print(prediction)

