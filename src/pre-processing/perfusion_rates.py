# Define the materials with their respective W and rho values
materials = {
    'bone': {'W': 0.12, 'rho': 1908},
    'fat': {'W': 1.1, 'rho': 911},
    'muscle': {'W': 3.6, 'rho': 1090},
    'tumor': {'W': 1.8, 'rho': 1090}
}

# Calculate the W/rho ratio for each material
ratios = {material: data['W'] / data['rho'] for material, data in materials.items()}

# Sort the materials by their W/rho ratio from smallest to largest
sorted_ratios = sorted(ratios.items(), key=lambda item: item[1])

# Display the results
print("Material Ratios (W/rho) from Smallest to Largest:")
for material, ratio in sorted_ratios:
    print(f"{material.capitalize()}: {ratio:.6f}")