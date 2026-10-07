import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_phase_4_silver_aristocracy():
    # ---------------------------------------------------------
    # 1. Global Parameters (Phase 4 Logic)
    # ---------------------------------------------------------
    n_agents = 5000
    T = 100            # 100-year Post-New Deal Era
    dt = 1/12          # Monthly steps
    steps = int(T / dt)
    np.random.seed(42)
    
    # Endowments (The Inheritance Gap)
    # High SES: $500 (Latent Heat / Buffer)
    # Low SES: $0 (The Sub-Critical Floor)
    w = np.concatenate([np.full(n_agents//2, 500.0), np.full(n_agents//2, 0.0)])
    is_high_ses = np.concatenate([np.ones(n_agents//2, dtype=bool), np.zeros(n_agents//2, dtype=bool)])
    
    # Demographics
    age = np.random.randint(0, 85, n_agents).astype(float)
    unemp_timer = np.zeros(n_agents)
    
    # Child Tracking (Max 3 Children per household)
    n_children = np.zeros(n_agents, dtype=int)
    child_timers = [np.zeros(n_agents) for _ in range(3)] 
    
    # Economic Constants
    salary_yr = 55.0        # Labor Income
    col_adult_yr = 50.0     # Baseline "Eating" Cost
    child_cost_yr = 25.0    # Fixed 50% "Child Tax"
    job_penalty = 10.0      # Restart Haircut (delta)
    parent_threshold = 375.0 # Required wealth (3yrs CoL)
    
    # Market Risk/Return
    mu_cap = 0.08           # 8% "Gifted" Potential
    sigma_cap = 0.25        # 25% "Ferrari" Volatility
    shock_prob = 0.0137     # Calibrated for 4% steady-state unemployment
    unemployment_duration = 3 
    
    # Storage for time-series visualization
    results_median = []
    results_avg = []
    
    # ---------------------------------------------------------
    # 2. Main Simulation Loop
    # ---------------------------------------------------------
    for t in range(steps):
        age += dt
        
        # A. Death & Dynastic Replacement (Inheritance)
        dead = (age >= 85)
        if np.any(dead):
            # The next generation begins - High SES inherit, Low SES reset to 0
            w[dead & is_high_ses] = 500.0
            w[dead & ~is_high_ses] = 0.0
            age[dead] = 0
            n_children[dead] = 0
            for i in range(3): child_timers[i][dead] = 0
            unemp_timer[dead] = 0

        # B. Labor Market (The Effort Bucket)
        is_worker = (age >= 20) & (age < 70)
        is_retiree = (age >= 70) & (age < 85)
        
        # Unemployment shocks
        new_shock = (np.random.rand(n_agents) < shock_prob) & (unemp_timer == 0) & is_worker
        unemp_timer[new_shock] = unemployment_duration
        
        salary_mo = np.zeros(n_agents)
        salary_mo[is_worker & (unemp_timer == 0)] = salary_yr * dt
        
        # Apply Job Penalty (Restart Cost) at the moment of re-employment
        just_hired = (unemp_timer == 1)
        w[just_hired] -= job_penalty
        unemp_timer[unemp_timer > 0] -= 1
        
        # C. The Sandwich Transfer (10% Wage tax from workers to retirees)
        total_transfer_pool = np.sum(salary_mo) * 0.10
        salary_mo[is_worker] *= 0.90 
        
        num_retirees = np.sum(is_retiree)
        if num_retirees > 0:
            w[is_retiree] += (total_transfer_pool / num_retirees)
            
        # D. Capital Growth (The Luck Bucket)
        z = np.random.normal(0, 1, n_agents)
        growth = (mu_cap - 0.5 * sigma_cap**2) * dt + sigma_cap * np.sqrt(dt) * z
        w[w > 0] *= np.exp(growth[w > 0])
        
        # E. The Parent Trap (Reproduction Mechanic)
        can_have_kid = (age >= 25) & (age < 45) & (w >= parent_threshold) & (n_children < 3)
        # Random birth chance if threshold is met
        new_kid = can_have_kid & (np.random.rand(n_agents) < (1/48))
        for i in np.where(new_kid)[0]:
            idx = n_children[i]
            child_timers[idx][i] = 18.0
            n_children[i] += 1
        
        # F. Fixed Drains (Eating + Children)
        active_child_costs = np.zeros(n_agents)
        for i in range(3):
            is_active = child_timers[i] > 0
            active_child_costs[is_active] += (child_cost_yr * dt)
            child_timers[i][is_active] -= dt
        
        w += salary_mo - (col_adult_yr * dt) - active_child_costs
        
        # G. The Ruin Barrier (Volatility Trap)
        w[w < 0.01] = 0.01 
        
        results_median.append(np.median(w))
        results_avg.append(np.mean(w))
        
    # ---------------------------------------------------------
    # 3. Visualization
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    time = np.linspace(0, T, steps)
    
    # Figure 1: Average vs Median
    ax1.plot(time, results_avg, label='Average Wealth (Ensemble)', color='green', linewidth=2)
    ax1.plot(time, results_median, label='Median Wealth (Typical Agent)', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Figure 1: 100-Year Wealth Divergence (Post-New Deal Era)', fontweight='bold')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Wealth ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 2: Log-Quintile Distribution
    retiree_w = w[(age >= 70) & (age < 85)]
    parent_w = w[(age >= 25) & (age < 45)]

    def get_quintile_means(data):
        if len(data) == 0: return [0.01]*5
        data_sorted = np.sort(data)
        chunks = np.array_split(data_sorted, 5)
        return [np.mean(c) for c in chunks]

    p_quint = get_quintile_means(parent_w)
    r_quint = get_quintile_means(retiree_w)
    
    x = np.arange(5)
    width = 0.35
    ax2.bar(x - width/2, p_quint, width, label='Parents (Ages 25-45)', color='navy', alpha=0.8)
    ax2.bar(x + width/2, r_quint, width, label='Retirees (Ages 70-85)', color='darkorange', alpha=0.8)
    ax2.set_title('Figure 2: Wealth Concentration by Quintile (Log Scale)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax2.set_yscale('log')
    ax2.set_ylabel('Wealth ($)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

# Execution
run_phase_4_silver_aristocracy()
