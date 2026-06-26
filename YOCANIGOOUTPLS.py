import numpy as np

#making this so i know if i should even bother asking to go out or not.

def markov_canigoout_analysis(transition_matrix, initial_state, n_days,
                             state_names=("go", "dont go", "go, but get grounded", "go, but still get grounded"), verbose=True):
  
    P = np.array(transition_matrix, dtype=float)
    state = np.array(initial_state, dtype=float)
    n_states = P.shape[0]
    
    if P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square (NxN).")
    if state.shape[0] != n_states:
        raise ValueError("Initial state length must match the matrix size.")
    if len(state_names) != n_states:
        raise ValueError("state_names length must match the matrix size.")
    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Each row of the transition matrix must sum to 1.")
    if not np.isclose(state.sum(), 1):
        raise ValueError("Initial state probabilities must sum to 1.")
    
    if verbose:
        print(f"Transition Matrix (P):\n{P}\n")
        print(f"Initial state (Day 0): "
              f"{dict(zip(state_names, np.round(state, 4)))}\n")
        print("Possibility forecast:")
 
    current = state.copy()
    for day in range(1, n_days + 1):
        current = current @ P
        if verbose:
            probs = dict(zip(state_names, np.round(current, 4)))
            print(f"  Day {day}: {probs}")
 
    # Long-run (steady-state) distribution, found via eigenvectors of P^T
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)]
    stationary = np.real(stationary[:, 0])
    stationary = stationary / stationary.sum()
 
    if verbose:
        print(f"\nLong-run (steady-state) distribution: "
              f"{dict(zip(state_names, np.round(stationary, 4)))}")
 
    return current
 
 
if __name__ == "__main__":
    transition_matrix = [
        [0.6, 0.2, 0.1, 0.1], #go out fine
        [0.5, 0.1, 0.4, 0.0], #dont go out
        [0, 1, 0, 0], #go out cooked
        [0.0, 0.2, 0.6, 0.2], #dont go out but still cooked
    ]
 
    initial_state = [1, 0, 0, 0]   # I'm going today we ball
    n_days = 2                # How cooked am I tomorrow
 
    final_probs = markov_canigoout_analysis(
        transition_matrix, initial_state, n_days
    )
 
    final_probs = markov_canigoout_analysis(transition_matrix, initial_state, n_days)
 
    print(f"\n>>> Final probabilities after {n_days} days:")
    for name, p in zip(("1", "2", "3", "4"), final_probs):
        print(f"    P({name}) = {p:.4f}")