from ortools.linear_solver import pywraplp
from time import time
solver = pywraplp.Solver.CreateSolver("SCIP") # Create the mip solver

# Sample data
data = {}
data["weights"] = [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36] # Weight of each item
data["values"] = [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25] # Value of each item
assert len(data["weights"]) == len(data["values"])

data["num_items"] = len(data["weights"])
data["all_items"] = range(data["num_items"])

data["vehicle_limits"] = [50, 100, 32, 120] # Weight limit of each vehicle
data["num_vehicles"] = len(data["vehicle_limits"])
data["all_vehicles"] = range(data["num_vehicles"])



# Variables
x = {} # x[i, b] = 1 if item i is packed in vehicle v. 0 otherwise. 
for i in data["all_items"]:
    for v in data["all_vehicles"]:
        x[i, v] = solver.BoolVar(f"x_{i}_{v}")

# Constraints
# Each item is assigned to at most one vehicle
for i in data["all_items"]:
    solver.Add(sum(x[i, b] for b in data["all_vehicles"]) <= 1)

# The weight packed in each vehicle cannot exceed its weight limit
for v in data["all_vehicles"]:
    solver.Add(
        sum(x[i, v] * data["weights"][i] for i in data["all_items"])
        <= data["vehicle_limits"][v]
    )

# Objective
# Maximize total value of packed items.
objective = solver.Objective()
for i in data["all_items"]:
    for v in data["all_vehicles"]:
        objective.SetCoefficient(x[i, v], data["values"][i])
objective.SetMaximization()

# Solve & print solution
start_Time = time()
status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    total_weight = 0
    for v in data["all_vehicles"]:
        print(f"🚚 Vehicle {v} (Weight Limit: {data['vehicle_limits'][v]})")
        vehicle_weight = 0
        vehicle_value = 0
        for i in data["all_items"]:
            if x[i, v].solution_value() > 0:
                print(
                    f" 📦 Item {i} weight: {data['weights'][i]} value: {data['values'][i]}"
                )
                vehicle_weight += data["weights"][i]
                vehicle_value += data["values"][i]
        print(f" - Packed vehicle weight: {vehicle_weight}")
        print(f" - Packed vehicle value: {vehicle_value}\n")
        total_weight += vehicle_weight
    print(f"Total packed weight: {total_weight}")
    print(f"\n💰Total packed value: ${objective.Value():.2f}M")
else:
    print("🚨 The problem does not have an optimal solution.")

print(f"That only took {(time() - start_Time):.3f}s")