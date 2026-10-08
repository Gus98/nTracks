import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def R_i_inv(x, N, omega_A, omega_D):
    return (1/2)* ( x/N + 1 + omega_D/(N*omega_A) - np.sqrt((x/N + 1 + omega_D/(N*omega_A))**2 - 4*x/N))
    # \frac{1}{2}\left(\frac{x}{N}+1+\frac{\alpha}{N\beta}-
    # \sqrt{\left(\frac{x}{N}+1+\frac{\alpha}{N\beta}\right)^{2}-\frac{4x}{N}}\right)

def solve_steady_state(alpha=0.9, beta=0.4, delta = 0.01, N = 3.0, omega_A = 0.3, omega_D = 1.0):
    T = N*omega_A + omega_D 
    # A little janky, but this repurposes the function to pass in total concentration values
    # alpha = R_i_inv(alpha, N, omega_A, omega_D)
    # beta = 1-R_i_inv(beta, N, omega_A, omega_D)

    print("Solving steady-state for parameters: ")
    print(f"Parameters: alpha={alpha}, beta={beta}, delta={delta}, N={N}, omega_A={omega_A}, omega_D={omega_D}, T={T}")
    phase = ""
    if alpha >1/2:
        if beta > 1/2:
            phase = "MC"
            bulk = 1/2
        else:
            phase = "HD"
            bulk = alpha
    else:
        if alpha > beta:
            phase = "HD"
            bulk = alpha
        else:
            phase = "LD"
            bulk = 1-beta
    print(f"Predicted phase: {phase}")
    # 1. Define the mesh (x-coordinates)
    x = np.linspace(0, 1, 10) # Start with a coarse mesh
    
    # 2. Define the system of ODEs: y' = f(x, y)
    # y[0] is u, y[1] is u'
    def odes(x, y):
        u = y[0]
        up = y[1]
        dudx = up
        dupdx = 2.0/delta * up *(1.0 - 2.0*u)  # The operator L(u) logic
        return np.vstack((dudx, dupdx))

    # 3. Define the Boundary Conditions (residuals)
    # ya: values at x[0], yb: values at x[-1]
    def bc(ya, yb):
        return np.array([ya[0] - alpha,  # u(0) = alpha
                         yb[0] - (1-beta)]) # u(1) = 1-beta

    # 4. Provide an initial guess for y
    y_guess = np.zeros((2, x.size))
    #y_guess[0,:] = np.full(x.size, bulk) # this actually makes some of them break idk why

    # 5. Solve
    res = solve_bvp(odes, bc, x, y_guess)

    # 6. Plotting
    x_plot = np.linspace(0, 1, 100)
    y_plot = res.sol(x_plot)[0] # Extract u(x)
    # Transform to total concentration
    off_plot = (omega_D*y_plot)/(omega_A * (1-y_plot))

    return x_plot, y_plot, off_plot

if __name__ == "__main__":
    #print(solve_steady_state())
    #alpha=1.0
    #T = N*alpha + beta  
    #print(solve_steady_state())
    #alpha = 5.0
    #T = N*alpha + beta
    #print(solve_steady_state())


    # 1. Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Convert the 2x2 matrix into a 1D list for easy looping

    # 2. Parameters to iterate over
    # Each tuple corresponds to (A, B, delta, N, alpha, beta)
    k_values = [(0.2,0.75,0.02,1.0,1.0,0.9), 
                (0.8,0.35,0.02,1.0,1.0,0.9), 
                (0.8,0.75,0.02,1.0,1.0,0.9), 
                (0.2,0.3,0.02,1.0,1.0,0.9)]

    # 3. Loop through parameters and axes simultaneously
    for i, k in enumerate(k_values):
        ax = axes[i]
        x_vals, u_vals, off_vals = solve_steady_state(*k)
        T = k[3]*k[4] + k[5]
        alpha = k[4]
        # Plot on the specific subplot
        ax.plot(x_vals, u_vals, color='crimson', lw=2)
        ax.plot(x_vals, off_vals, color='green', lw=2)
        ax.set_title(f"alpha={k[0]}, beta={k[1]}, epsilon={k[2]}, N={k[3]}, omega_A={k[4]}, omega_D={k[5]}", fontsize=12)
        ax.set_title("")
        ax.set_xlabel("x")
        ax.set_ylabel("p(x)")
        ax.set_ylim(0,1)
        #ax.axhline(k[3]/2 + k[4]/k[5], color='blue', linestyle=":", label='MC Value')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', lw=0.5)

    # 4. Cleanup and Show
    plt.suptitle("Steady-state Analysis for TASEP \n (first two parameters are A, B)", fontsize=16)
    plt.tight_layout() # Prevents titles and labels from overlapping
    plt.show()

