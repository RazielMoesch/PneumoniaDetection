import os
from inference import PneumoniaPredictor


###Example of using an image path
#If doesn't work make sure your in the correct directory
pneumonia_positive_path = "test_images/Pneumonia_False.png"
#instanciate predictor
predictor = PneumoniaPredictor()
#predict and graph
prediction = predictor.predict(pneumonia_positive_path, graph_pred=True)
#print
print(prediction)

