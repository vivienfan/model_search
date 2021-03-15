import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

def preprocessing():
	data=pd.read_excel('credit_default.xls',skiprows=1)
	data=data.rename({'default payment next month':'default'}, axis=1)
	categorical=['SEX',	'EDUCATION',	'MARRIAGE']
	numeric=['PAY_0',	'PAY_2',	'PAY_3',	'PAY_4',	'PAY_5',	'PAY_6']
	data_final=pd.get_dummies(
		data=data[categorical], 
		columns=['SEX',	'EDUCATION',	'MARRIAGE'])
	data_default=pd.concat([data['default'], data_final, data[numeric]],  axis=1)
	print(data_default['default'].value_counts())
	data_default.to_csv('default.csv', sep = ',', index = False, header = None)

def train():
	trainer = single_trainer.SingleTrainer(
		data=csv_data.Provider(
			label_index=0, 
			logits_dimension=2, 
			record_defaults=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
			filename="default.csv"),
		spec='../../model_search/configs/dnn_config.pbtxt')
	trainer.try_models(
    number_models=20,
    train_steps=200,
    eval_steps=1,
    root_dir="./",
    batch_size=512,
    experiment_name="example2",
    experiment_owner="model_search_user")

def main(argv):
	preprocessing()
	train()

if __name__ == '__main__':
	app.run(main)