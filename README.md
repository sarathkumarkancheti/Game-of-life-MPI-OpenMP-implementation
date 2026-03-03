# Game-of-life-MPI-OpenMP-implementation
# HPC Project: Game of Life on a Hexahedral Mesh

This project implements a parallelized version of the "Game of Life" evolved on a hexagonal grid rather than a standard square checkerboard.

## 🧬 Simulation Rules
The simulation follows the standard Game of Life logic but is adapted for a **hexahedral mesh** where each cell has only **6 neighbors**.

* **Birth**: A dead cell with exactly 3 neighbors is born in the next step.
* **Survival**: A living cell with 2 or 3 neighbors survives.
* **Death (Loneliness)**: A cell with 0 or 1 neighbor dies.
* **Death (Overpopulation)**: A cell with more than 3 neighbors dies.



## 💻 Parallel Architecture
The application uses a **hybrid MPI/OpenMP** approach to ensure high performance and scalability on large boards.

* **MPI (Distributed Memory)**: The fields are distributed **row-wise** among processes.
* **Memory Management**: The complete board is **never** stored on the master rank; data is distributed across all ranks to save memory.
* **OpenMP (Shared Memory)**: Utilized to accelerate loops within each MPI process[cite: 23].
* **Boundaries**: Supports both dead-cell boundaries and a **two-dimensional torus** (circular) configuration.

## 🚀 Performance Evaluation
* Scalability is a core requirement of this project[cite: 26]. The following metrics are analyzed in the accompanying documentation:
* **Strong Scaling**: Increasing the number of ranks (cores) for a fixed problem size.
* **Weak Scaling**: Increasing the number of ranks proportionally to the problem size.

> **Note**: Output is suppressed during scalability testing to ensure accurate measurements.

## 🛠️ Usage
### Compilation
Ensure you have an MPI-enabled compiler (like `mpicc`) and OpenMP flags enabled.
```bash
mpicc -fopenmp main.c -o game_of_life
# To run with P MPI processes and T OpenMP threads:

export OMP_NUM_THREADS=T
mpirun -np P ./game_of_life
