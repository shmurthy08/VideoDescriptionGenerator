import os
import datetime

'''
File will create a way to visualize the tensorboard outputs
'''

inputs = input("Do you want to start tensorboard? (y/n): ")
if inputs == 'n':
    exit(0)
elif inputs != 'y':
    print("Invalid input")
    exit(0)
# Start tensorboard
else: 
    os.system(f'tensorboard --logdir=logs/hparam_tuning20240503-203819/ --bind_all')
