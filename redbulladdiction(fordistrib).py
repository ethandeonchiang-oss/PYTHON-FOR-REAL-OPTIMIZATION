import numpy as np

# writing this code as a practical and personal demostration of the the demand costs of items
# here I document how often and when I do purchase Red Bull (sadly not sponsored), and how much on avg it runs me yearly

# MODULE 1: Yearly Simulation of Red Bull Consumption (base)

def simulate_one_year(rop=2, pack_size=4, cost_pack=345):
    lam_normal = -np.log(3/7)  
    lam_exam = -np.log(1/7)    
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    daily_lambdas = np.full(365, lam_normal)
    day_to_month = []
    for m_idx, num_days in enumerate(days_in_months):
        day_to_month.extend([m_idx + 1] * num_days)
        
    for day in range(365):
        m = day_to_month[day]
        if m in [1, 2]: daily_lambdas[day] = lam_normal + (lam_exam - lam_normal) * (day / 59.0)
        elif m == 3:    daily_lambdas[day] = lam_exam
        elif m in [10, 11]: daily_lambdas[day] = lam_normal + (lam_exam - lam_normal) * ((day - 273) / 61.0)
        elif m == 12:   daily_lambdas[day] = lam_exam

    inventory, open_orders = 4, 0
    purchase_days = [0] 
    
    daily_inventory = np.zeros(365)
    daily_spending = np.zeros(365)
    daily_stockouts = np.zeros(365)
    daily_spending[0] = cost_pack
    
    for day in range(365):
        recent_purchases = 0
        for p_day in reversed(purchase_days):
            if day - p_day < 7: recent_purchases += 1
            else: break
                
        max_packs_allowed = 2 if daily_lambdas[day] > 1.4 else 1
        can_buy_more = recent_purchases < max_packs_allowed
        
        day_demand = min(np.random.poisson(daily_lambdas[day]), 2)
        
        if inventory >= day_demand: inventory -= day_demand
        else:
            daily_stockouts[day] = 1 
            inventory = 0
            
        if inventory <= rop and can_buy_more:
            packs_to_buy = 2 if (daily_lambdas[day] > 1.5 and inventory == 0 and recent_purchases == 0) else 1
            daily_spending[day] = packs_to_buy * cost_pack
            purchase_days.extend([day] * packs_to_buy) 
            inventory += (packs_to_buy * pack_size)
            
        daily_inventory[day] = inventory
        
    return daily_inventory, daily_spending, daily_stockouts


# MODULE 2: Monte Carlo Simulation of Red Bull Consumption (averaging over many years)

def run_monte_carlo(simulations=20000, rop=2):
    """Runs the 1-year simulation thousands of times and averages the results."""
    print(f"Running {simulations:,} simulations for ROP={rop}...")
    
    all_inv = np.zeros((simulations, 365))
    all_spend = np.zeros((simulations, 365))
    all_stockouts = np.zeros((simulations, 365))

    for i in range(simulations):
        inv, spend, stock = simulate_one_year(rop=rop)
        all_inv[i, :] = inv
        all_spend[i, :] = spend
        all_stockouts[i, :] = stock

    return (np.mean(all_inv, axis=0), 
            np.mean(all_spend, axis=0), 
            np.mean(all_stockouts, axis=0))


def print_reports(avg_inv, avg_spend, avg_stockout_prob):
    """Formats and prints the Weekly, Monthly, and Yearly data."""
    
    print("\n=== CLEAN CALENDAR MONTHLY BREAKDOWN ===")
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    print(f"{'Month':<15}{'Avg End Inv':<16}{'Total Spend':<16}{'Expected Days Wiped Out':<25}")
    print("-" * 75)
    
    current_idx = 0
    for m, num_days in enumerate(days_in_months):
        m_slice = slice(current_idx, current_idx + num_days)
        m_inv = avg_inv[current_idx + num_days - 1]
        m_spend = np.sum(avg_spend[m_slice])
        m_stockouts = np.sum(avg_stockout_prob[m_slice])
        
        tag = " (EXAMS)" if (m + 1) in [3, 12] else ""
        print(f"{month_names[m]+tag:<15}{m_inv:<16.2f}Php {m_spend:<12.2f}{m_stockouts:<25.2f}")
        current_idx += num_days

    print("\n=== YEARLY CONSOLIDATED SUMMARY ===")
    print(f"Total Expected Red Bull Expenditure:    Php {np.sum(avg_spend):,.2f}")
    print(f"Total Days Spent Completely Wiped Out:  {np.sum(avg_stockout_prob):.1f} days / 365")
    print("==========================================\n")




def main():
    print("Red Bull Consumption Analysis...")
    
    # Test an ROP of 2
    inv_2, spend_2, stock_2 = run_monte_carlo(simulations=20000, rop=2)
    print_reports(inv_2, spend_2, stock_2)

# Call the function to trigger the whole script
main()