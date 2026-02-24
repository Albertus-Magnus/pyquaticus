from pyquaticus.utils.rewards import triple_aggressive_rew, triple_caps_and_grabs
import numpy as np
import random

#########################
# Q-learning own implementation to work with pyquaticus. Requires some representation of the state space and action space, as well as a reward function (or compatibility with pyquaticus reward functions). This will for now be a very basic implementation, without any opponent agents or multi-agent considerations. Just to get a working example of Q-learning in a simple environment (hopefully avoiding state space explosion). 
# Later we will extend it to opponent agents. Custom reward is also later, first use reward that is written down in training data.
#########################

# Build a q-table from a file of info and reward (etc?) logs and safe that table (is now a q-learn policy)
class QPyquaBuilder:
    def __init__(self, x=160, y=80, x_cells=8, y_cells=8):
        '''
            x, y describe the map size. 
            x_cells is the number of cell columns in the grid divided by two (per map-half).
            y_cells is the number of cell rows in the grid.
        '''
        # As first step we must divide the map into a grid of cells.
        # The map is 160 by 80, the longer side (x-axis) is divided into the own half and the enemy half, where cells should only be located on one half at a time.
        #worldgrid = np.array([
        # Divide the map into a grid of cells, assuming x-achsis is divided into two team sides:
        self.x_cellsize = (x / 2) / x_cells # zB: 160 / 2 / 8 = 10
        self.y_cellsize = y / y_cells # zB: 80 / 8 = 10
        # Cellsize optimally should align with the border between map halves.
        self.cellgrid = np.zeros((x_cells, y_cells)) # This is just a grid to represent the map-dividing cells, not the actual q-table. At least for each represented agent position we need one of those for the full (but still oversimplified) state.
        self.oobool = False # Necessary if position temporarily leaves the grid. Stored in addition to the map, because probably(?) only one state necessary.
        
        ''' The Q-table needs account for 
                - the position of the agent (for which the rewards are tabled).
                - the positions of 2 (3?) opponents.
                - 
            (Probably store the qtable as this-class-object for now?, not too much in table form)
        '''