"""
This script is adapted from the article: "Constrained Optimization for Decision Making in Health Care Using Python: A Tutorial"
Published on: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10625722/

Author(s): K. H. Benjamin Leung, Nasrin Yousefi, Timothy C. Y. Chan, and Ahmed M. Bayoumi

Original code and methodology are used under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
For details, see: https://creativecommons.org/licenses/by/4.0/

Please refer to the original article for detailed methodology and results.
"""

# Import necessary packages
import gurobipy as gp
import pandas as pd
import yaml
from gurobipy import GRB

# Load the data set that contains distances between
# cardiac arrests and potential AED locations
covered = pd.read_csv("Example2_distanceMatrix.csv", index_col="ID")

# PARAMETERS
# Load dynamic input
with open("input.yaml", "r") as file:
    params = yaml.safe_load(file)

# Coverage cutoff limit
coverage_distance = params["coverage_distance"]

# Number of AEDs to be placed
K = params["K"]

# Compute a_ij, which states whether each cardiac arrest is
# within the coverage distance of each potential AED location
covered = (covered <= coverage_distance).astype(int)
n_cases = covered.shape[0]
n_candidates = covered.shape[1]
A = pd.DataFrame.to_numpy(covered)

# Create a Gurobi model object
m = gp.Model("AED")

# DECISION VARIABLES
# x is whether an AED is placed at potential AED location j
x = m.addMVar(n_candidates, vtype=GRB.BINARY, name="x")

# y is whether cardiac arrest i is covered by at least one AED
y = m.addMVar(n_cases, vtype=GRB.BINARY, name="y")

# CONSTRAINTS
# Cardiac arrest covered if at least one AED is
# placed within range
m.addConstr(A @ x >= y)

# Total number of AEDs placed
m.addConstr(x.sum() == K)

# OBJECTIVE
# Maximize the number of covered cardiac arrests
m.setObjective(y.sum(), gp.GRB.MAXIMIZE)

# Run the optimization model
m.optimize()

# Print the optimal solution and its objective value
selected = []
for j in x.tolist():
    if j.X > 0.5:
        selected.append(j.index)
print("Selected locations: ", end="")
print(*selected, sep=", ")
print("Number of covered cardiac arrests: " + str(int(m.objVal)))
