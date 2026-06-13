from pathlib import Path

# Root directory
root = Path(r"C:\Users\zhich\OneDrive - جامعة أبي بكر بلقايد تلمسان\April 2025\Experiments")

# List of directories to create
directories = [
    root / "FOA_V0",
    root / "FOA_Memory_V2",
    root / "Baselines" / "GWO",
    root / "Baselines" / "PSO",
    root / "Baselines" / "Random",
    root / "Baselines" / "Greedy",
    root / "Comparative_Study",
]

# Create directories
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created: {directory}")

print("\nDirectory structure created successfully.")