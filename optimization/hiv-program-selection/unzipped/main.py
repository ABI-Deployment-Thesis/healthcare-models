"""
This script is adapted from the article: "Constrained Optimization for Decision Making in Health Care Using Python: A Tutorial"
Published on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10625722/

Author(s): K. H. Benjamin Leung, Nasrin Yousefi, Timothy C. Y. Chan, and Ahmed M. Bayoumi

Original code and methodology are used under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
For details, see: https://creativecommons.org/licenses/by/4.0/

Please refer to the original article for detailed methodology and results.
"""

# Import necessary packages
import json

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# Create a Gurobi model
m = gp.Model("HIV")

# PARAMETERS
# Load dynamic input
with open("input.json", "r") as file:
    params = json.load(file)

# Costs, per person reached, by each program
# Cost values are divided by 1000 for computational reasons
c = np.array(params["c"])

# QALYs gained, per person reached, by each program
a = np.array(params["a"])

# Minimum/maximum number of persons reached by each program
# in array form
x_min = np.array(params["x_min"])
x_max = np.array(params["x_max"])

# Budget
# The budget is divided by 1000 for computational reasons
B = params["B"]

# DECISION VARIABLES
# The number of people to reach with each program
x = m.addMVar(3, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="Program")

# CONSTRAINT
# The total cost across all programs must equal the budget
m.addConstr(c @ x == B)

# OBJECTIVE
# Maximize the total number of QALYs gained
m.setObjective(a @ x, GRB.MAXIMIZE)

# Run the optimization model
m.optimize()

# Print the optimal solution and its objective value
for v in m.getVars():
    print("%s = %.4f" % (v.varName, v.x))
print("Max QALYs = %.4f" % m.objVal)
