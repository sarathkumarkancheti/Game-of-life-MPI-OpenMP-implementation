/* 
 * ============================================================================
 * Game of Life on Hexahedral Mesh - Hybrid MPI/OpenMP Implementation
 * ============================================================================
 * 
 * This program implements Conway's Game of Life on a hexagonal grid using
 * MPI for distributed memory parallelism and OpenMP for shared memory parallelism.
 * 
 * Compilation:
 *   mpicc -fopenmp -O3 game_of_life.c -o game_of_life.out -lm
 * 
 * Execution:
 *   mpirun -np 4 ./game_of_life 1000 1000 50
 *   Arguments: num_rows num_cols num_iterations
  * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/* ========================================================================== */

#define TRUE 1
#define FALSE 0
typedef int bool;

/* Cell states */
#define DEAD 0
#define ALIVE 1

/* Boundary condition types */
#define TORUS 1
#define DEAD_BORDER 0

/* Global configuration */
typedef struct {
    int global_rows;        /* Total number of rows in grid */
    int global_cols;        /* Total number of columns in grid */
    int num_iterations;     /* Number of time steps to simulate */
    int boundary_type;      /* TORUS (1) or DEAD_BORDER (0) */
    int verbose;            /* 1: print output, 0: silent */
} Config;

/* MPI process variables */
typedef struct {
    int rank;               /* MPI rank of current process */
    int num_ranks;          /* Total number of MPI processes */
    int local_rows;         /* Rows owned by this rank */
    int local_cols;         /* Columns (same for all ranks) */
    int row_start;          /* Global row index where local data starts */
    int row_end;            /* Global row index where local data ends */
    int boundary_type;      /* TORUS or DEAD_BORDER */
} ProcessInfo;

/* Game grid with ghost cells */
typedef struct {
    /* Current state: local_rows + 2 ghost rows, local_cols columns */
    unsigned char **current;
    /* Next state  */
    unsigned char **next;
    /* Local dimensions including ghost cells */
    int rows_with_ghost;
    int cols;
    /* Pointers to send/receive buffers */
    unsigned char *send_top;
    unsigned char *send_bottom;
    unsigned char *recv_top;
    unsigned char *recv_bottom;
} Grid;

/* ========================================================================== */
/*                                FUNCTIONS                                   */
/* ========================================================================== */

/**
 * Allocate memory for 2D array
 */
unsigned char** allocate_2d_array(int rows, int cols) {
    unsigned char **array = (unsigned char **)malloc(rows * sizeof(unsigned char *));
    unsigned char *data = (unsigned char *)calloc(rows * cols, sizeof(unsigned char));
    
    if (array == NULL || data == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        array[i] = &data[i * cols];
    }
    return array;
}

/**
 * Free 2D array allocated by allocate_2d_array
 */
void free_2d_array(unsigned char **array) {
    if (array != NULL) {
        free(array[0]);  /* Free contiguous data */
        free(array);     /* Free row pointers */
    }
}

/**
 * Initialize grid with random cell states
 * 50% probability of each cell being alive
 */
void initialize_grid(Grid *grid, ProcessInfo *pinfo, unsigned int seed) {
    srand(seed + pinfo->rank);  /* Different starting point per rank */
    
    int local_rows = grid->rows_with_ghost - 2;  /* Exclude ghost rows */
    
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < grid->cols; j++) {
            grid->current[i][j] = (rand() % 2 == 0) ? ALIVE : DEAD;
        }
    }
    
    /* Ghost rows initialized to DEAD by calloc */
}

/**
 * Print text form of the grid for debugging
 * Only print first/last few rows to limit output
 */
void print_grid(Grid *grid, ProcessInfo *pinfo, int iteration) {
    if (pinfo->rank != 0) return;
    
    printf("\n===== Iteration %d =====\n", iteration);
    
    int local_rows = grid->rows_with_ghost - 2;
    int print_rows = (local_rows > 10) ? 10 : local_rows;
    
    for (int i = 1; i <= print_rows; i++) {
        printf("Row %d: ", pinfo->row_start + i - 1);
        for (int j = 0; j < grid->cols && j < 50; j++) {
            printf("%c", grid->current[i][j] ? '*' : '.');
        }
        if (grid->cols > 50) printf("...");
        printf("\n");
    }
}

/* ========================================================================== */
/*                         HEXAGONAL GRID FUNCTIONS                           */
/* ========================================================================== */

/**
 * Count living neighbors of a cell in hexagonal grid
 * Column wraparound is handled by count_neighbors for TORUS mode
 *            Row wraparound is handled by exchange_ghost_cells
 */
int count_neighbors(Grid *grid, ProcessInfo *pinfo, int row, int col) {
    int count = 0;
    int global_row = pinfo->row_start + (row - 1);  /* Adjust for ghost row offset */
    
    /* Neighbor offsets (depend on whether row is even/odd) */
    int even_offsets[][2] = {
        {-1, -1}, {-1, 0}, {0, -1}, {0, 1}, {1, -1}, {1, 0}
    };
    
    int odd_offsets[][2] = {
        {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, 0}, {1, 1}
    };
    
    int (*offsets)[2] = (global_row % 2 == 0) ? even_offsets : odd_offsets;
    
    for (int i = 0; i < 6; i++) {
        int nr = row + offsets[i][0];
        int nc = col + offsets[i][1];
        
        /* Column boundaries with torus wrapping */
        if (nc < 0 || nc >= grid->cols) {
            if (pinfo->boundary_type == TORUS) {
                nc = (nc + grid->cols) % grid->cols;
            } else {
                continue;  /* Skip neighbor outside boundary */
            }
        }
        
        /* Handle row boundaries */
        if (nr < 0 || nr >= grid->rows_with_ghost) {
            continue;  /* Ghost cells handle this */
        }
        
        if (grid->current[nr][nc] == ALIVE) {
            count++;
        }
    }
    
    return count;
}

/* ========================================================================== */
/*                       MPI COMMUNICATION FUNCTIONS                          */
/* ========================================================================== */

/**
 * Exchange ghost cells (boundary rows) with neighboring MPI processes
 * 
 * TORUS MODE:
 * - First rank (rank 0) receives from last rank (rank N-1)
 * - Last rank (rank N-1) receives from first rank (rank 0)
 * - Creates a circular topology 
 * 
 * DEAD_BORDER MODE:
 * - Ghost rows at boundaries remain DEAD
 * - No wraparound communication
 */
void exchange_ghost_cells(Grid *grid, ProcessInfo *pinfo, MPI_Comm comm) {
    MPI_Request requests[4];
    int request_count = 0;
    int tag_top = 0;
    int tag_bottom = 1;
    int local_rows = grid->rows_with_ghost - 2;  /* Exclude ghost rows */
    
    /* Data for sending */
    memcpy(grid->send_top, grid->current[1], grid->cols * sizeof(unsigned char));
    memcpy(grid->send_bottom, grid->current[local_rows], grid->cols * sizeof(unsigned char));
    
    if (pinfo->boundary_type == TORUS) {
        /* TORUS TOPOLOGY: All ranks wrap around  */
       
        /* Rank above this rank (wraps around if this is rank 0) */
        int rank_above = (pinfo->rank == 0) ? pinfo->num_ranks - 1 : pinfo->rank - 1;
        
        /* Rank below this rank (wraps around if this is last rank) */
        int rank_below = (pinfo->rank == pinfo->num_ranks - 1) ? 0 : pinfo->rank + 1;
        
        /* Send top row to rank above, receive into top ghost from rank above */
        MPI_Isend(grid->send_top, grid->cols, MPI_UNSIGNED_CHAR,
                  rank_above, tag_bottom, comm, &requests[request_count++]);
        MPI_Irecv(grid->recv_top, grid->cols, MPI_UNSIGNED_CHAR,
                  rank_above, tag_top, comm, &requests[request_count++]);
        
        /* Send bottom row to rank below, receive into bottom ghost from rank below */
        MPI_Isend(grid->send_bottom, grid->cols, MPI_UNSIGNED_CHAR,
                  rank_below, tag_top, comm, &requests[request_count++]);
        MPI_Irecv(grid->recv_bottom, grid->cols, MPI_UNSIGNED_CHAR,
                  rank_below, tag_bottom, comm, &requests[request_count++]);
        
    } else {
        /* DEAD_BORDER MODE: Boundary ranks don't communicate */
        /* Ghost cells remain DEAD (initialized to 0 by calloc) */
        
        /* Send/receive with rank above (if exists) */
        if (pinfo->rank > 0) {
            MPI_Isend(grid->send_top, grid->cols, MPI_UNSIGNED_CHAR,
                      pinfo->rank - 1, tag_bottom, comm, &requests[request_count++]);
            MPI_Irecv(grid->recv_top, grid->cols, MPI_UNSIGNED_CHAR,
                      pinfo->rank - 1, tag_top, comm, &requests[request_count++]);
        } else {
            /* First rank: no rank above, keep ghost row as DEAD */
            memset(grid->recv_top, DEAD, grid->cols * sizeof(unsigned char));
        }
        
        /* Send/receive with rank below (if exists) */
        if (pinfo->rank < pinfo->num_ranks - 1) {
            MPI_Isend(grid->send_bottom, grid->cols, MPI_UNSIGNED_CHAR,
                      pinfo->rank + 1, tag_top, comm, &requests[request_count++]);
            MPI_Irecv(grid->recv_bottom, grid->cols, MPI_UNSIGNED_CHAR,
                      pinfo->rank + 1, tag_bottom, comm, &requests[request_count++]);
        } else {
            /* Last rank: no rank below, keep ghost row as DEAD */
            memset(grid->recv_bottom, DEAD, grid->cols * sizeof(unsigned char));
        }
    }
    
    /* Wait for all communications to complete */
    if (request_count > 0) {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }
    
    /* Copy received data to ghost cells */
    memcpy(grid->current[0], grid->recv_top, grid->cols * sizeof(unsigned char));
    memcpy(grid->current[local_rows + 1], grid->recv_bottom, grid->cols * sizeof(unsigned char));
}

/* ========================================================================== */
/*                       GAME OF LIFE UPDATE FUNCTION                         */
/* ========================================================================== */

/**
 * Perform one time step of Game of Life on hexagonal grid
 * 
 * Rules:
 * - A dead cell with exactly 3 neighbors becomes alive
 * - A living cell with 2-3 neighbors survives
 * - All other cells die or stay dead
 */
void update_grid(Grid *grid, ProcessInfo *pinfo) {
    int local_rows = grid->rows_with_ghost - 2;
    
    /* Process each cell (excluding ghost rows) */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 0; j < grid->cols; j++) {
            int neighbors = count_neighbors(grid, pinfo, i, j);
            unsigned char current_state = grid->current[i][j];
            
            /* Apply Game of Life rules */
            if (current_state == ALIVE) {
                /* Living cell */
                grid->next[i][j] = (neighbors == 2 || neighbors == 3) ? ALIVE : DEAD;
            } else {
                /* Dead cell */
                grid->next[i][j] = (neighbors == 3) ? ALIVE : DEAD;
            }
        }
    }
    
    /* Swap grid pointers (next becomes current) */
    unsigned char **temp = grid->current;
    grid->current = grid->next;
    grid->next = temp;
    
    /* Copy non-ghost rows back to next for next iteration */
    #pragma omp parallel for
    for (int i = 1; i <= local_rows; i++) {
        memcpy(grid->next[i], grid->current[i], grid->cols * sizeof(unsigned char));
    }
}

/* ========================================================================== */
/*                          MAIN SIMULATION                                   */
/* ========================================================================== */

void run_simulation(Grid *grid, ProcessInfo *pinfo, Config *config, MPI_Comm comm) {
    double start_time = MPI_Wtime();
    
    for (int iter = 0; iter < config->num_iterations; iter++) {
        /* Exchange boundary data with neighbors */
        exchange_ghost_cells(grid, pinfo, comm);
        
        /* Synchronize all processes */
        MPI_Barrier(comm);
        
        /* Update grid according to Game of Life rules */
        update_grid(grid, pinfo);
        
        /* Print grid periodically for debugging */
        if (config->verbose && iter % 10 == 0) {
            print_grid(grid, pinfo, iter);
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (pinfo->rank == 0) {
        printf("Simulation completed in %.3f seconds\n", end_time - start_time);
        printf("Performance: %.2f million cells/sec\n",
               (config->global_rows * config->global_cols * config->num_iterations) / 
               ((end_time - start_time) * 1e6));
    }
}

/* ========================================================================== */
/*                             MAIN PROGRAM                                   */
/* ========================================================================== */

int main(int argc, char *argv[]) {
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    /*  command line arguments */
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <num_rows> <num_cols> <num_iterations>\n", argv[0]);
            fprintf(stderr, "Example: mpirun -np 4 %s 10000 10000 100\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int global_rows = atoi(argv[1]);
    int global_cols = atoi(argv[2]);
    int num_iterations = atoi(argv[3]);
    
    /* Check inputs */
    if (global_rows <= 0 || global_cols <= 0 || num_iterations <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: All parameters must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (global_rows < num_ranks) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of rows must be >= number of MPI ranks\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    /* Configuration */
    Config config;
    config.global_rows = global_rows;
    config.global_cols = global_cols;
    config.num_iterations = num_iterations;
    config.boundary_type = TORUS;              /* ← TORUS or DEAD_BORDER */
    config.verbose = (num_iterations <= 20) ? TRUE : FALSE;  /* Only prints for small iterations */
    
    /* Process information */
    ProcessInfo pinfo;
    pinfo.rank = rank;
    pinfo.num_ranks = num_ranks;
    pinfo.local_cols = global_cols;
    pinfo.boundary_type = config.boundary_type;
    
    /* Distribute rows among processes */
    int base_rows = global_rows / num_ranks;
    int extra_rows = global_rows % num_ranks;
    
    pinfo.local_rows = base_rows + (rank < extra_rows ? 1 : 0);
    pinfo.row_start = rank * base_rows + (rank < extra_rows ? rank : extra_rows);
    pinfo.row_end = pinfo.row_start + pinfo.local_rows;
    
    if (rank == 0) {
        printf("========================================\n");
        printf("Game of Life on Hexahedral Mesh (MPI/OpenMP)\n");
        printf("========================================\n");
        printf("Grid size: %d x %d\n", global_rows, global_cols);
        printf("Iterations: %d\n", num_iterations);
        printf("MPI processes: %d\n", num_ranks);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Boundary Condition: %s\n", 
               config.boundary_type == TORUS ? "TORUS" : "DEAD_BORDER");
        printf("========================================\n\n");
    }
    
    /* Allocate grid data */
    Grid grid;
    grid.cols = global_cols;
    grid.rows_with_ghost = pinfo.local_rows + 2;  /* +2 for top and bottom ghost rows */
    
    grid.current = allocate_2d_array(grid.rows_with_ghost, grid.cols);
    grid.next = allocate_2d_array(grid.rows_with_ghost, grid.cols);
    grid.send_top = (unsigned char *)malloc(grid.cols * sizeof(unsigned char));
    grid.send_bottom = (unsigned char *)malloc(grid.cols * sizeof(unsigned char));
    grid.recv_top = (unsigned char *)malloc(grid.cols * sizeof(unsigned char));
    grid.recv_bottom = (unsigned char *)malloc(grid.cols * sizeof(unsigned char));
    
    if (grid.send_top == NULL || grid.send_bottom == NULL || 
        grid.recv_top == NULL || grid.recv_bottom == NULL) {
        fprintf(stderr, "Rank %d: Memory allocation failed for communication buffers\n", rank);
        MPI_Finalize();
        return 1;
    }
    
    /* Initialize grid with random configuration */
    initialize_grid(&grid, &pinfo, (unsigned int)time(NULL));
    
    /* Run simulation */
    run_simulation(&grid, &pinfo, &config, MPI_COMM_WORLD);
    
    /* Print final */
    int total_alive = 0;
    for (int i = 1; i < grid.rows_with_ghost - 1; i++) {
        for (int j = 0; j < grid.cols; j++) {
            if (grid.current[i][j] == ALIVE) {
                total_alive++;
            }
        }
    }
    
    int global_alive = 0;
    MPI_Reduce(&total_alive, &global_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double density = (double)global_alive / (global_rows * global_cols) * 100.0;
        printf("Final live cells: %d (%.2f%% density)\n", global_alive, density);
    }
    
    /* Cleaning */
    free_2d_array(grid.current);
    free_2d_array(grid.next);
    free(grid.send_top);
    free(grid.send_bottom);
    free(grid.recv_top);
    free(grid.recv_bottom);
    
    MPI_Finalize();
return 0;
}





