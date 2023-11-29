# -----------------------------------------------------------------------------#                                                                   #
# @date     june 12, 2021                                                      #
# @brief    test semantics                                                     #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")

    
def is_inside(aabb1, aabb2):
    a1 = np.array(aabb1.center - aabb1.sizes / 2)
    b1 = np.array(aabb1.center + aabb1.sizes / 2)
    
    a2 = np.array(aabb2.center - aabb2.sizes / 2)
    b2 = np.array(aabb2.center + aabb2.sizes / 2)
    
    return (
        np.all(a1 <= a2) and np.all(b1 >= a2) and 
        np.all(a1 <= b2) and np.all(b1 >= b2)
    )

def run_exp(state, regions):
    pos = state.position
    agent_region = None
    for region in regions:
        name = region.category.name()
        A = np.array(region.aabb.center - (region.aabb.sizes / 2))
        B = np.array(region.aabb.center + (region.aabb.sizes / 2))
        if np.all(A <= pos) and np.all(pos <= B):
            if agent_region == None:
                agent_region = region
               
            else:
                if is_inside(region.aabb, agent_region.aabb):
                    agent_region = region
    if agent_region:
        return agent_region.category.name()
    else :
        return 'no label'

