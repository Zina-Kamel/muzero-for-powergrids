"""
SharedStorage is where Network objects are stored. This is utilised when running mutliple threads in the training.
After a certain thread finishes collecting data about the env, they can retrieve the last latest Network to 
train and store back for other threads to pick  

Currently the implementation runs sequentially so we always have one network object.
"""

from network import PowerGridNetwork

class SharedStorage(object):
    
    def __init__(self, config):
        self.network = PowerGridNetwork(config)

    def get_last_network(self):
        return self.network
