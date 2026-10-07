import numpy as np
import matplotlib.pyplot as plt

def run_comparative_tax_simulation():
    # ---------------------------------------------------------
    # 1. Global Parameters
    # ---------------------------------------------------------
    n_agents = 5000
    T = 100            # 100 Years
    dt = 1/12          # Monthly time steps
    steps = int(T / dt)
    
    # Common Economic Parameters
    salary_yr = 55.0        
    col_adult_yr = 50.0     
    child_cost_yr = 25.0    
    job_penalty = 10.0      
    parent_threshold = 375.0 
    mu_cap = 0.08           # 8% Return (The "Ferrari")
    sigma_cap = 0.25        # 25% Volatility
    shock_prob = 0.0137     # 4% Unemployment Rate
    unemployment_duration = 3 
    
    # The Two Scenarios
    scenarios = [
        {
            "name": "Scenario A: Annual Tax (30% Cap / 0% Estate)", 
            "labor_tax": 0.10, 
            "annual_cap_tax": 0.30, 
            "estate_tax": 0.0
        },
        {
            "name": "Scenario B: Deferred Tax (0% Cap / 50% Estate)", 
            "labor_tax": 0.10, 
            "annual_cap_tax": 0.0, 
            "estate_tax": 0.50
        }
    ]
    
    results = {}

    print(f"Running Simulation for {T} Years with N={n_agents} Agents...")

    for sc in scenarios:
        # Reset Agents
        # High SES start at $500; Low SES start at $0
        w = np.concatenate([np.full(n_agents//2, 500.0), np.full(n_agents//2, 0.0)])
        is_high_ses = np.concatenate([np.ones(n_agents//2, dtype=bool), np.zeros(n_agents//2, dtype=bool)])
        
        # Consistent demographics reset
        np.random.seed(42) 
        age = np.random.randint(0, 85, n_agents).astype(float)
        unemp_timer = np.zeros(n_agents)
        n_children = np.zeros(n_agents, dtype=int)
        child_timers = [np.zeros(n_agents) for _ in range(3)] 
        
        median_history = []
        aggregate_history = []
        
        # Re-seed for loop dynamics
        np.random.seed(42) 

        for t in range(steps):
            age += dt
            
            # --- DEATH & ESTATE TAX (The Harvest) ---
            dead = (age >= 85)
            estate_tax_revenue = 0.0
            
            if np.any(dead):
                dead_wealth = w[dead].copy()
                
                # Apply Estate Tax
                tax_collected = dead_wealth * sc["estate_tax"]
                estate_tax_revenue = np.sum(tax_collected)
                
                # Inheritance (Heir gets the remainder)
                w[dead] -= tax_collected
                
                # Reset Demographics for Heir
                age[dead] = 0
                n_children[dead] = 0
                for i in range(3): child_timers[i][dead] = 0
                unemp_timer[dead] = 0

            # --- LABOR MARKET ---
            is_worker = (age >= 20) & (age < 70)
            
            new_shock = (np.random.rand(n_agents) < shock_prob) & (unemp_timer == 0) & is_worker
            unemp_timer[new_shock] = unemployment_duration
            
            salary_mo = np.zeros(n_agents)
            salary_mo[is_worker & (unemp_timer == 0)] = salary_yr * dt
            
            just_hired = (unemp_timer == 1)
            w[just_hired] -= job_penalty
            unemp_timer[unemp_timer > 0] -= 1
            
            # --- LABOR TAX ---
            tax_labor_rev = np.sum(salary_mo * sc["labor_tax"])
            salary_mo *= (1 - sc["labor_tax"])
            
            # --- CAPITAL GROWTH & ANNUAL TAX ---
            current_w = w.copy()
            z = np.random.normal(0, 1, n_agents)
            growth = (mu_cap - 0.5 * sigma_cap**2) * dt + sigma_cap * np.sqrt(dt) * z
            w[w > 0] *= np.exp(growth[w > 0])
            
            # Annual Capital Tax (Only for Scenario A)
            annual_tax_rev = 0.0
            if sc["annual_cap_tax"] > 0:
                gains = w - current_w
                taxable_gains = np.maximum(gains, 0)
                tax_collected = taxable_gains * sc["annual_cap_tax"]
                annual_tax_rev = np.sum(tax_collected)
                w -= tax_collected
            
            # --- DIVIDEND (The Mechanism) ---
            # Pool all three sources: Labor Tax + Annual Cap Tax + Estate Tax
            total_revenue = tax_labor_rev + annual_tax_rev + estate_tax_revenue
            dividend = total_revenue / n_agents
            w += dividend
            
            # --- COSTS ---
            active_child_costs = np.zeros(n_agents)
            for i in range(3):
                is_active = child_timers[i] > 0
                active_child_costs[is_active] += (child_cost_yr * dt)
                child_timers[i][is_active] -= dt
            
            w += salary_mo - (col_adult_yr * dt) - active_child_costs
            
            # Floor (Absorbing Barrier)
            w[w < 0.01] = 0.01 
            
            # Reproduction
            can_have_kid = (age >= 25) & (age < 45) & (w >= parent_threshold) & (n_children < 3)
            new_kid = can_have_kid & (np.random.rand(n_agents) < (1/48))
            for i in np.where(new_kid)[0]:
                idx = n_children[i]
                child_timers[idx][i] = 18.0
                n_children[i] += 1
            
            median_history.append(np.median(w))
            aggregate_history.append(np.sum(w))
            
        results[sc["name"]] = {"median": median_history, "aggregate": aggregate_history}

    # ---------------------------------------------------------
    # 2. Visualization
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    time = np.linspace(0, T, steps)
    
    # Plot Aggregate Wealth
    for name, data in results.items():
        color = 'red' if "Annual" in name else 'green'
        style = '--' if "Annual" in name else '-'
        ax1.plot(time, data["aggregate"], label=name, color=color, linestyle=style, linewidth=2)
    
    ax1.set_title('Figure 1: Aggregate Wealth (The Pie)', fontweight='bold')
    ax1.set_ylabel('Total Wealth ($) - Log Scale')
    ax1.set_xlabel('Years')
    ax1.set_yscale('log') # Log scale needed to see the magnitude difference
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot Median Wealth
    for name, data in results.items():
        color = 'red' if "Annual" in name else 'green'
        style = '--' if "Annual" in name else '-'
        ax2.plot(time, data["median"], label=name, color=color, linestyle=style, linewidth=2)
    
    ax2.set_title('Figure 2: Median Wealth (The Household)', fontweight='bold')
    ax2.set_ylabel('Median Wealth ($)')
    ax2.set_xlabel('Years')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print Final Stats
    print("\nFinal Statistics (Year 100):")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Aggregate Wealth: ${data['aggregate'][-1]:,.0f}")
        print(f"  Median Wealth:    ${data['median'][-1]:,.0f}")
        print("-" * 30)

run_comparative_tax_simulation()
