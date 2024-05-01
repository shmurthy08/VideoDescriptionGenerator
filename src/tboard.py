import os
'''
File will create a way to visualize the tensorboard outputs
'''
inputs = input("Do you want to start tensorboard? (y/n): ")
if inputs == 'n':
    exit(0)
    # Destroy any existing tensorboard processes
    os.system('killall tensorboard')
elif inputs != 'y':
    print("Invalid input")
    exit(0)
# Start tensorboard
else: 
    os.system('tensorboard --logdir=logs/fit')

