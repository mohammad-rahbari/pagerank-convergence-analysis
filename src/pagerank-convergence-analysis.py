import numpy as np
import matplotlib.pyplot as plt

# PageRank matrix M with damping factor m
def calculate_M(A, m=0.15):
    n = A.shape[0]
    M = np.zeros((n, n))
    for j in range(n):
        col_sum = np.sum(A[:, j])
        if col_sum == 0:  # dangling node
            M[:, j] = 1.0 / n
        else:
            M[:, j] = (1 - m) * (A[:, j] / col_sum) + (m / n)
    return M

# Power iteration to find stationary vector q
def power_iteration(M, tol=1e-10, max_iter=1000):
    n = M.shape[0]
    v = np.ones(n) / n
    for i in range(max_iter):
        v_new = M @ v
        if np.linalg.norm(v_new - v, 1) < tol:
            return v_new, i
        v = v_new
    return v, max_iter

# Load hollins.dat
def load_hollins(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    n_nodes, n_edges = int(lines[0].split()[0]), int(lines[0].split()[1]) 
    A = np.zeros((n_nodes, n_nodes)) #creat  6012*6012 matrix
    for i in range(n_nodes + 1, len(lines)):
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            src, tgt = int(parts[0]) - 1, int(parts[1]) - 1 # Each line "src tgt" means: page src links to page tgt

            A[tgt, src] = 1
    return A, n_nodes

# Exercise 14: Error analysis
def exercise_14(M, name=""):
    print(f"\n{'='*80}")
    print(f"Exercise 14: {name}")

    
    n = M.shape[0]
    
    # Initial guess x_0: not too close to q (use e_1 basis vector)
    x_0 = np.zeros(n)
    x_0[0] = 1.0  # Start with all probability on first page, not uniform distribution
    
    # Compute stationary vector q
    q, final_k = power_iteration(M)
    print(f"q converged in {final_k} iterations")
    
    # Required k values from problem + final iteration for best λ₂ estimate
    required_k = [1, 5, 10, 50]
    # Include final_k only if it's different from required values
    k_values = sorted(set(required_k + [final_k]))
    
    # Compute errors ||M^k x_0 - q||_1 up to max needed iteration
    max_k = max(k_values)
    errors = {}
    x_k = x_0.copy()
    for k in range(max_k + 1):
        errors[k] = np.linalg.norm(x_k - q, 1)
        x_k = M @ x_k
    
    print(f"\nk\t||M^k x_0 - q||_1\tRatio (error_k/error_k-1)")
    print("-" * 55)
    for k in k_values:
        ratio = errors[k] / errors[k-1] if k > 0 and errors[k-1] > 0 else "-"
        if isinstance(ratio, float):
            print(f"{k}\t{errors[k]:.10e}\t{ratio:.10f}")
        else:
            print(f"{k}\t{errors[k]:.10e}\t{ratio}")
    
    # Show all errors to prove monotonic decrease
    print(f"\nAll errors (showing monotonic decrease):")
    print("-" * 40)
    for k in range(0, max(k_values) + 1):
        if k <= 15 or k in k_values:
            print(f"k={k:2d}: error = {errors[k]:.10e}")
    
    # Theoretical bound c = max_j |1 - 2*min_i M_ij|
    c = max(abs(1 - 2 * np.min(M[:, j])) for j in range(n))
    print(f"\nc = {c:.10f}")
    
    # Second largest eigenvalue |lambda_2| - compute directly
    # For large matrices, we compute eigenvalues of a smaller matrix or use theoretical bound
    if n <= 1000:  # Direct computation feasible for small matrices
        eigenvalues = np.linalg.eigvals(M)
        eigenvalues_abs = np.abs(eigenvalues)
        eigenvalues_sorted = np.sort(eigenvalues_abs)[::-1]  # Sort descending
        lambda_2 = eigenvalues_sorted[1]  # Second largest
        print(f"|lambda_2| = {lambda_2:.10f}")
    else:
        # For large matrices (like Hollins), use theoretical upper bound
        lambda_2 = 1 - 0.15  # = 0.85 for m=0.15
        print(f"|lambda_2| <= {lambda_2:.10f} (theoretical upper bound)")
        
    
    # Verification
    print(f"\nVerification: |lambda_2| < c ? {lambda_2 < c} ({lambda_2:.6f} < {c:.6f})")
    
    print(f"\nk\tRatio\t\tRatio < c?")
    print("-" * 40)
    for k in k_values:
        if k > 0 and errors[k-1] > 0:
            ratio = errors[k] / errors[k-1]
            print(f"{k}\t{ratio:.6f}\t{ratio < c}")
    
    return errors, c, lambda_2, final_k



#Exersice 11
#Link matrix from exercise 11
A11 = np.array([[0, 0, 1/2, 1/2, 0],
              [1/3, 0, 0, 0, 0], 
              [1/3, 1/2, 0, 1/2, 1],
              [1/3, 1/2, 0, 0, 0],
              [0,0,1/2,0, 0]])  

#calculate matrix M from A11
M11 = calculate_M(A11, m=0.15)
print("\nCalculating PageRank matrix M for Exercise 11...")
print(f"Matrix M11:\n{M11}")

#calculate eigvector PageRank for M11
eigenvector11, iters11 = power_iteration(M11)
print(f"\nExercise 11: Calculated PageRank vector for M11 in {iters11} iterations:")
print("eigenvector M11 =", eigenvector11)
print(f"\nSum of PageRank vector for M11: {eigenvector11.sum():.10f}")

# Run on Hollins dataset
print("\nLoading hollins.dat...")
A_hollins, n = load_hollins("..\\data\\hollins.dat")
print(f"Loaded: {n} nodes")




#Exersice 14
print("\nCalculating PageRank matrix M...")
M_hollins = calculate_M(A_hollins, m=0.15) #calculating matrix M 
print(f"Matrix M: {M_hollins.shape}, column sums: [{np.sum(M_hollins, axis=0).min():.4f}, {np.sum(M_hollins, axis=0).max():.4f}]")
#calculate errors, ratios, 
errors, c, lambda_2, final_k = exercise_14(M_hollins, name="Hollins Dataset")




# PLOTTING RESULTS

max_k_values = [50, 10, 5, 1]
colors = ['blue', 'orange', 'green', 'red']

# Plot 1: Error per iteration over matrix
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for max_k, color in zip(max_k_values, colors):
    k_range = list(range(max_k + 1))
    err_range = [errors[k] for k in k_range]
    ax1.plot(k_range, err_range, '-o', color=color, markersize=4, label=f'Max K: {max_k}')

ax1.set_xlabel('Iteration k')
ax1.set_ylabel('Error')
ax1.set_title(f'Error per iteration over {M_hollins.shape[0]}x{M_hollins.shape[1]} Google matrix')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error ratio per iteration over matrix
for max_k, color in zip(max_k_values, colors):
    k_range = list(range(1, max_k + 1))
    ratio_range = [errors[k]/errors[k-1] for k in k_range]
    ax2.plot(k_range, ratio_range, '-o', color=color, markersize=4, label=f'Max K: {max_k}')

ax2.set_xlabel('Iteration k')
ax2.set_ylabel('Error Ratio (error_k / error_{k-1})')
ax2.set_title(f'Error ratio per iteration over {M_hollins.shape[0]}x{M_hollins.shape[1]} Google matrix')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise14_plots.png', dpi=150)
plt.show()

print("\nPlot saved: exercise14_plots.png")

print("\nPlots saved: plot1_error_per_iteration.png, plot2_ratio_per_iteration.png")

# Find k with best |lambda_2| estimate (in numerically stable range 10^-6 to 10^-4)
# Use the last k in the stable range 
stable_k_values = [k for k in range(1, final_k+1) if 1e-6 <= errors[k] <= 1e-4]
if stable_k_values:
    best_k = max(stable_k_values)  # Last k in stable range (most converged)
    print(f"Best |lambda_2| estimate at k={best_k}: ratio = {errors[best_k]/errors[best_k-1]:.10f}")



# Exercise 14 Applied to Exercise 11 Web (5-page network)

print("="*80)
print("\n--- Direct Eigenvalue Analysis for 5-page Web ---")

# Use the M11 matrix computed earlier for Exercise 11
errors_ex11, c_ex11, lambda_2_ex11, final_k_ex11 = exercise_14(M11, name="Exercise 11 Web (5 pages)")

# direct eigenvalue computation for small matrix
eigenvalues_M11 = np.linalg.eigvals(M11)
eigenvalues_M11_sorted = np.sort(np.abs(eigenvalues_M11))[::-1]
print(f"Eigenvalues of M11 (absolute values, sorted descending):")
for i, ev in enumerate(eigenvalues_M11_sorted):
    print(f"  |lambda_{i+1}| = {ev:.10f}")

# compute actual second largest eigenvalue
actual_lambda2_ex11 = eigenvalues_M11_sorted[1]
print(f"\nActual |lambda_2| = {actual_lambda2_ex11:.10f}")
print(f"Theoretical c = {c_ex11:.10f}")
print(f"Verification: |lambda_2| < c ? {actual_lambda2_ex11 < c_ex11} ({actual_lambda2_ex11:.6f} < {c_ex11:.6f})")

# summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON: Hollins Dataset vs Exercise 11 Web")
print(f"{'Metric':<35} {'Hollins (6012 pages)':<25} {'Ex11 Web (5 pages)':<25}")
print("-"*85)
print(f"{'Iterations to converge':<35} {final_k:<25} {final_k_ex11:<25}")
print(f"{'c = max|1-2*min M_ij|':<35} {c:.10f}           {c_ex11:.10f}")
print(f"{'|lambda_2| (second eigenvalue)':<35} {'<= 0.85 (theoretical)':<25} {actual_lambda2_ex11:.10f}")
print(f"{'Final error':<35} {errors[final_k]:.6e}           {errors_ex11[final_k_ex11]:.6e}")