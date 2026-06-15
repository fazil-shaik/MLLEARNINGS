import numpy as np
import pandas as pd

def generate_coffee_dataset(n_samples=1000):
    """Generate synthetic coffee roasting data"""
    np.random.seed(42)
    
    # Generate features
    roast_time = np.random.uniform(8, 18, n_samples)
    temp_ramp = np.random.uniform(3, 8, n_samples)
    moisture = np.random.uniform(8, 14, n_samples)
    density = np.random.uniform(0.6, 0.8, n_samples)
    airflow = np.random.randint(1, 10, n_samples)
    
    # Target variables with realistic relationships
    # Acidity decreases non-linearly with roast time
    acidity = 10 - 0.8*roast_time + 0.03*roast_time**2 + np.random.normal(0, 0.5, n_samples)
    acidity = np.clip(acidity, 1, 10)
    
    # Sweetness peaks at medium roasts (polynomial relationship)
    sweetness = -0.05*(roast_time - 12)**2 + 8 + np.random.normal(0, 0.5, n_samples)
    sweetness = np.clip(sweetness, 1, 10)
    
    # Body influenced by moisture and density
    body = 3 + 0.5*moisture + 5*density + np.random.normal(0, 0.5, n_samples)
    body = np.clip(body, 1, 10)
    
    df = pd.DataFrame({
        'roast_time_min': roast_time,
        'temp_ramp_c_min': temp_ramp,
        'moisture_pct': moisture,
        'density_g_ml': density,
        'airflow': airflow,
        'acidity': acidity,
        'sweetness': sweetness,
        'body': body
    })
    
    return df

# Generate and save
df_synthetic = generate_coffee_dataset(5000)
df_synthetic.to_csv('coffee_synthetic.csv', index=False)