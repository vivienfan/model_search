import sys 
sys.path.insert(0, '../../')
from absl import app
from absl import flags
FLAGS = flags.FLAGS 
FLAGS(sys.argv) 

import model_search
from model_search import constants
from model_search import single_trainer
from model_search.data import csv_data

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def main(argv):
  trainer = single_trainer.SingleTrainer(
    data=csv_data.Provider(
      label_index=0,
      logits_dimension=2,
      record_defaults=[0, 0, 0, 0],
      filename="../../model_search/data/testdata/csv_random_data.csv"),
    spec="../../model_search/configs/dnn_config.pbtxt")
  trainer.try_models(
    number_models=200,
    train_steps=1000,
    eval_steps=100,
    root_dir="./",
    batch_size=32,
    experiment_name="example0",
    experiment_owner="model_search_user")

if __name__ == '__main__':
  app.run(main)