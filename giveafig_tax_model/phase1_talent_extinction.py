import numpy as np
import matplotlib.pyplot as plt

def run_phase_1_final_three_panels_v2():
    # Parameters
    n_agents = 5000 
    T = 40
    dt = 0.1
    steps = int(T / dt)
    
    W0 = 1000.0
    W_barrier = 200.0  # Ruin Barrier
    
    # Groups: (Name, Mu, Sigma, Color, Style)
    # Using distinct colors/styles to ensure all 4 are visible
    groups = [
        ("Normie / High SES", 0.05, 0.10, "tab:blue", "-"),      # Safe / Safe
        ("Normie / Low SES", 0.05, 0.20, "tab:cyan", "--"),      # Safe / Risky
        ("Gifted / High SES", 0.08, 0.20, "tab:green", "-"),     # Risky / Safe
        ("Gifted / Low SES", 0.08, 0.40, "tab:red", "--")        # Risky / Risky
    ]
    
    # Store data
    results_avg = {}
    results_median = {}
    results_survival = {}
    
    np.random.seed(42)
    
    for name, mu, sigma, color, style in groups:
        wealth = np.full(n_agents, W0)
        alive = np.ones(n_agents, dtype=bool)
        
        avg_hist = [W0]
        median_hist = [W0]
        survival_hist = [1.0]
        
        for t in range(steps):
            if np.any(alive):
                z = np.random.normal(0, 1, n_agents)
                growth = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                
                # Update wealth for alive agents
                wealth[alive] *= np.exp(growth[alive])
                
                # Check Ruin
                ruined = (wealth < W_barrier) & alive
                alive[ruined] = False
                wealth[ruined] = 0.0
            
            avg_hist.append(np.mean(wealth))
            median_hist.append(np.median(wealth))
            survival_hist.append(np.mean(alive))
            
        results_avg[name] = avg_hist
        results_median[name] = median_hist
        results_survival[name] = survival_hist

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    time = np.linspace(0, T, steps + 1)
    
    # Common function to plot with distinct styles
    def plot_lines(ax, data_dict):
        for name, data in data_dict.items():
            # Retrieve style info
            grp = next(g for g in groups if g[0] == name)
            color = grp[3]
            style = grp[4]
            # Plot
            ax.plot(time, data, label=name, color=color, linestyle=style, linewidth=2.5, alpha=0.9)

    # Panel 1: Average
    plot_lines(ax1, results_avg)
    ax1.set_title('Average Wealth (The Illusion)', fontweight='bold')
    ax1.set_ylabel('Wealth ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='small')
    
    # Panel 2: Median
    plot_lines(ax2, results_median)
    ax2.set_title('Median Wealth (The Reality)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize='small')
    
    # Panel 3: Survival
    # Multiply by 100 for percentage
    results_survival_pct = {k: np.array(v)*100 for k, v in results_survival.items()}
    plot_lines(ax3, results_survival_pct)
    ax3.set_title('Survival Rate (The Mechanism)', fontweight='bold')
    ax3.set_ylabel('% Surviving')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left', fontsize='small')
    
    plt.tight_layout()
    plt.show()

run_phase_1_final_three_panels_v2()
