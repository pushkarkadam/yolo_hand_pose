import sys 
sys.path.append('..')

from yolov1 import *


model_config = 'config.yaml'

model = ModelTrain(model_config)

model.fit()