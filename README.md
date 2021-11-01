# torch.save, torch.load

Post: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

## save checkpoint
```
python save_checkpoint.py
```

Import all necessary libraries for loading our data
Define and intialize the neural network
Initialize the optimizer
Save the general checkpoint

## load checkpoint
Load the general checkpoint
```
python load_checkpoint.py
```

## Disadvantage

The disadvantage of this approach is that the serialized data is bound to the specific classes and the exact directory structure used when the model is saved. The reason for this is because pickle does not save the model class itself. Rather, it saves a path to the file containing the class, which is used during load time. Because of this, your code can break in various ways when used in other projects or after refactors.

# save, load, state_dict

Post: https://pytorch.org/tutorials/beginner/saving_loading_models.html

How to run?

```
python main_state_dict.py
```

Output:

```
(hm) wxf@protago-hp01-3090:~/pytorch_prjs/test_centers/pytorch-checkpoint/save_state_dict$ python main_state_dict.py
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias       torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias       torch.Size([16])
fc1.weight       torch.Size([120, 400])
fc1.bias         torch.Size([120])
fc2.weight       torch.Size([84, 120])
fc2.bias         torch.Size([84])
fc3.weight       torch.Size([10, 84])
fc3.bias         torch.Size([10])
Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
saving...
saved...
loading...
loaded...
```