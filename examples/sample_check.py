import pandas 
import sys

sys.path.append('..')

from yolov1 import *

# Train 

df_train = pd.read_csv("../data/training.csv", index_col=False)

rendered_images, names = render_sample(df_train, '.', '', 'training')

plot_rendered_grid(rendered_images, names, dir_name='training_samples', path='../data')

# Evaluation 

df_eval = pd.read_csv("evaluation_both_hands.csv", index_col=False)

eval_rendered_images, eval_names = render_sample(df_eval, '../data/evaluation.csv', '', 'evaluation')

plot_rendered_grid(eval_rendered_images, eval_names, dir_name='evaluation_samples', path='../data')