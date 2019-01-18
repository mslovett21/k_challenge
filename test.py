from openeye import oechem
import os, sys
import re
import copy
import pandas as pd
import os
import numpy as np



# define 2D array to store the results
sim_matrix = np.zeros(shape=(41464,41464))


print("Hello")