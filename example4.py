import os
from inference import PneumoniaPredictor


###Example of using an image path
#If doesn't work make sure your in the correct directory
pneumonia_positive_path = "test_images/Pneumonia_True.png"
#instanciate predictor
predictor = PneumoniaPredictor()
#predict and graph
class_scores = predictor.predict(pneumonia_positive_path, graph_pred=True, return_class_scores=True, return_pred=False)
#print
print(class_scores)

