import numpy as np
import matplotlib.pyplot as plt

def run_phase_2_random_shocks():
    # Parameters
    n_agents = 5000
    T = 20  # Years
    dt = 1/12 # Monthly steps
    steps = int(T / dt)
    
    W0 = 1000.0
    W_barrier = 200.0
    
    # Common Baseline (Stable Times)
    mu_stable = 0.08
    sigma_stable = 0.25
    
    # Shock Parameters
    shock_prob = 1/12 # Probability per month
    
    # High SES (Buffered) Shock
    mu_shock_H = 0.00
    sigma_shock_H = 0.25
    
    # Low SES (Unbuffered) Shock
    mu_shock_L = -0.20
    sigma_shock_L = 0.45
    
    # Setup Groups
    groups = [
        ("High SES (Buffered)", "green", "-"),
        ("Low SES (Unbuffered)", "red", "--")
    ]
    
    # Data Storage
    results_avg = {g[0]: [W0] for g in groups}
    results_median = {g[0]: [W0] for g in groups}
    results_survival = {g[0]: [1.0] for g in groups}
    
    wealths = {
        groups[0][0]: np.full(n_agents, W0),
        groups[1][0]: np.full(n_agents, W0)
    }
    alives = {
        groups[0][0]: np.ones(n_agents, dtype=bool),
        groups[1][0]: np.ones(n_agents, dtype=bool)
    }
    
    np.random.seed(42)
    
    for t in range(steps):
        for name, color, style in groups:
            w = wealths[name]
            alive = alives[name]
            
            if np.any(alive):
                # Determine who gets shocked (Individual Shocks)
                is_shocked = np.random.rand(n_agents) < shock_prob
                
                # Parameters for each agent
                # Default to stable
                mus = np.full(n_agents, mu_stable)
                sigmas = np.full(n_agents, sigma_stable)
                
                # Apply shock params
                if name == "High SES (Buffered)":
                    mus[is_shocked] = mu_shock_H
                    sigmas[is_shocked] = sigma_shock_H
                else:
                    mus[is_shocked] = mu_shock_L
                    sigmas[is_shocked] = sigma_shock_L
                
                # Geometric Brownian Motion Step
                z = np.random.normal(0, 1, n_agents)
                growth = (mus - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * z
                
                # Update Wealth
                w[alive] *= np.exp(growth[alive])
                
                # Check Ruin
                ruined = (w < W_barrier) & alive
                alive[ruined] = False
                w[ruined] = 0.0
            
            results_avg[name].append(np.mean(w))
            results_median[name].append(np.median(w))
            results_survival[name].append(np.mean(alive))

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    time = np.linspace(0, T, steps + 1)
    
    def plot_panel(ax, data_dict, title, ylabel=None, ylim=None):
        for name, color, style in groups:
            y_vals = np.array(data_dict[name]) * 100 if "Survival" in title else data_dict[name]
            ax.plot(time, y_vals, label=name, color=color, linestyle=style, linewidth=2.5)
        
        ax.set_title(title, fontweight='bold')
        if ylabel: ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    plot_panel(ax1, results_avg, 'Average Wealth', ylabel='Wealth ($)')
    plot_panel(ax2, results_median, 'Median Wealth')
    plot_panel(ax3, results_survival, 'Survival Rate', ylabel='% Surviving', ylim=(0, 105))
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

run_phase_2_random_shocks()
