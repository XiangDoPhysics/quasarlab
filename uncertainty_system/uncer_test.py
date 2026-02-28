import qexpy as q

# Define measurements with their best estimates and uncertainties (e.g., Length = 0.76 +/- 0.15)
t = q.Measurement(0.76, 0.15)
x = q.Measurement(3, 0.1)

# Perform calculations using standard mathematical functions provided by QExpy
k = t / q.sqrt(x)

# Print the result with its uncertainty
print(k)