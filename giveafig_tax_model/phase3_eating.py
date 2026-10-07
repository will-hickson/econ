import numpy as np
import matplotlib.pyplot as plt

def run_phase_3_4pct_unemployment():
    # Parameters
    n_agents = 5000
    T = 100  # 100 Years
    dt = 1/12  # Monthly steps
    steps = int(T / dt)
    
    # Calibration for 4% Unemployment Rate:
    # If duration (D) = 3 months, and prob (p) is monthly:
    # Steady State Rate (u) = p*D / (1 + p*D - p)
    # 0.04 = 3p / (1 + 2p) -> 0.04 + 0.08p = 3p -> 0.04 = 2.92p -> p ≈ 0.0137
    shock_prob = 0.0137 
    unemployment_duration = 3 # months
    
    # Endowments (Latent Heat)
    w_high = np.full(n_agents, 500.0)
    w_low = np.full(n_agents, 50.0)
    
    # Economic Physics
    salary_yr = 55.0
    col_adult_yr = 50.0
    job_penalty = 10.0 # Restart penalty
    
    mu_cap = 0.08    # 8% annual return on savings
    sigma_cap = 0.25 # 25% annual volatility
    
    results = {
        "High SES": {"avg": [], "median": [], "survival": []},
        "Low SES": {"avg": [], "median": [], "survival": []}
    }
    
    np.random.seed(42)
    
    for name, wealth in [("High SES", w_high), ("Low SES", w_low)]:
        w = wealth.copy()
        alive = np.ones(n_agents, dtype=bool)
        unemp_timer = np.zeros(n_agents)
        
        for t in range(steps):
            if np.any(alive):
                # 1. Labor Market Dynamics
                # Check for new shocks
                new_shock = (np.random.rand(n_agents) < shock_prob) & (unemp_timer == 0)
                unemp_timer[new_shock] = unemployment_duration
                
                # Monthly Salary
                monthly_salary = np.full(n_agents, salary_yr * dt)
                monthly_salary[unemp_timer > 0] = 0.0
                
                # Apply Job Penalty at re-employment (end of month 3)
                just_hired = (unemp_timer == 1) 
                w[just_hired & alive] -= job_penalty
                
                # Decrement timer
                unemp_timer[unemp_timer > 0] -= 1
                
                # 2. Capital Growth (Multiplicative)
                z = np.random.normal(0, 1, n_agents)
                growth_mask = (w > 0) & alive
                growth = (mu_cap - 0.5 * sigma_cap**2) * dt + sigma_cap * np.sqrt(dt) * z
                w[growth_mask] *= np.exp(growth[growth_mask])
                
                # 3. Fixed Costs (Eating)
                current_col = col_adult_yr * dt
                w[alive] += monthly_salary[alive] - current_col
                
                # 4. Survival Check
                ruined = (w <= 0) & alive
                alive[ruined] = False
                w[ruined] = 0.0
                
            results[name]["avg"].append(np.mean(w))
            results[name]["median"].append(np.median(w))
            results[name]["survival"].append(np.mean(alive) * 100)

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    time = np.linspace(0, T, steps)
    
    for name, color, style in [("High SES", "green", "-"), ("Low SES", "red", "--")]:
        ax1.plot(time, results[name]["avg"], label=name, color=color, linestyle=style, linewidth=2)
        ax2.plot(time, results[name]["median"], label=name, color=color, linestyle=style, linewidth=2)
        ax3.plot(time, results[name]["survival"], label=name, color=color, linestyle=style, linewidth=2)
    
    ax1.set_title('Average Wealth (The Illusion)', fontweight='bold')
    ax2.set_title('Median Wealth (The Struggle)', fontweight='bold')
    ax3.set_title('Survival Rate (The Attrition)', fontweight='bold')
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax3.set_ylim(0, 105)
    plt.tight_layout()
    plt.show()

run_phase_3_4pct_unemployment()
