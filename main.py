
import sys
from preprocessing import Preprocessing
from network import My_Model
from train import Train
from test import Test
from feature import FeatureMappings

print(sys.argv)
if len(sys.argv) == 2 and sys.argv[1] == 'test':
    train_bool = False
else:
    train_bool = True


prep = Preprocessing()  
prep.create_iterators()  

feature_instance = FeatureMappings() 
train_dataset, val_dataset, test_dataset = feature_instance.create_features(prep)

network = My_Model()

for i in train_dataset:
    print(i[0].shape)
    print(i[1].shape)


if train_bool:
    # runs Train.py
    train_instance = Train(network, train_dataset, val_dataset)  
    train_instance.train()  #trains the model
else:
    print('test.......')
    test_obj = Test(network, test_dataset)
    test_obj.test() #tests the model
