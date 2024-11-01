import numpy as np
from pathlib import Path
import pandas as pd
from scipy import stats


def calculate_mae_uncertainty(experimental, theoretical, n_bootstrap=1000, confidence=0.95):
    """
    Calculate Mean Absolute Error (MAE) and its uncertainty using bootstrap resampling.
    
    Args:
        experimental (array-like): Experimental measurements
        theoretical (array-like): Theoretical predictions
        n_bootstrap (int): Number of bootstrap samples
        confidence (float): Confidence level for error bars (0 to 1)
        
    Returns:
        tuple: (mae, lower_error, upper_error)
            - mae: Mean Absolute Error
            - lower_error: Distance from MAE to lower confidence bound
            - upper_error: Distance from MAE to upper confidence bound
    """
    experimental = np.array(experimental)
    theoretical = np.array(theoretical)
    
    if len(experimental) != len(theoretical):
        raise ValueError("Experimental and theoretical arrays must have same length")
        
    # Calculate the base MAE
    mae = np.mean(np.abs(experimental - theoretical))
    
    # Perform bootstrap resampling
    bootstrap_maes = []
    n_points = len(experimental)
    
    for _ in range(n_bootstrap):
        # Generate random indices with replacement
        indices = np.random.randint(0, n_points, size=n_points)
        
        # Resample both arrays using the same indices to preserve pairing
        exp_sample = experimental[indices]
        theo_sample = theoretical[indices]
        
        # Calculate MAE for this bootstrap sample
        sample_mae = np.mean(np.abs(exp_sample - theo_sample))
        bootstrap_maes.append(sample_mae)
    
    # Calculate confidence intervals
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound, upper_bound = np.percentile(bootstrap_maes, [lower_percentile, upper_percentile])
    
    # Convert to error bar magnitudes
    lower_error = mae - lower_bound
    upper_error = upper_bound - mae
    
    # Calculate standard error for comparison
    std_error = np.std(bootstrap_maes)
    
    return {
        'mae': mae,
        'lower_error': lower_error,
        'upper_error': upper_error,
        'std_error': std_error,
        'bootstrap_distribution': bootstrap_maes
    }


def plot_mae_uncertainty(results):
    """
    Create a histogram of bootstrap results to visualize the uncertainty distribution.
    
    Args:
        results: Dictionary returned by calculate_mae_uncertainty
    """
    import matplotlib.pyplot as plt
    
    bootstrap_maes = results['bootstrap_distribution']
    mae = results['mae']
    
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_maes, bins=50, density=True, alpha=0.7)
    plt.axvline(mae, color='r', linestyle='--', label='MAE')
    plt.axvline(mae - results['lower_error'], color='g', linestyle=':', label='Confidence Interval')
    plt.axvline(mae + results['upper_error'], color='g', linestyle=':')
    
    # Fit and plot normal distribution for comparison
    mu, sigma = stats.norm.fit(bootstrap_maes)
    x = np.linspace(min(bootstrap_maes), max(bootstrap_maes), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', label='Normal Fit')
    
    plt.xlabel('MAE')
    plt.ylabel('Density')
    plt.title('Bootstrap Distribution of MAE')
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def write_results_to_file(results, processor: str, emitter: str, condition: str):
    if processor not in ("AQT", "QSCOUT"):
        raise ValueError(f"Invalid processor \"{processor}\"")
    
    file = Path(__file__).parent / "MAE.csv"
    if file.exists():
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame({"processor": [], "emitter": [], "condition": [], "MAE": [], "lower_err": [], "upper_err": []})

    df = df.set_index(["processor", "emitter", "condition"])
    df.loc[(processor, emitter, condition), ["MAE", "lower_err", "upper_err"]] = [
        results["mae"],
        results["lower_error"],
        results["upper_error"]
    ]
    df.to_csv(file)
