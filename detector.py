import json
import numpy as np
import joblib
from tensorflow import keras
from scipy.signal import welch
import collections
import time
import socket
import pandas as pd # <--- CHANGE 1: Import pandas

# --- 1. Helper Function for FFT (Identical to training) ---
def calculate_band_power(data, fs, freq_band):
    freqs, psd = welch(data, fs=fs, nperseg=len(data))
    band_indices = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]
    if len(band_indices) == 0:
        return 0
    return np.sum(psd[band_indices])

# --- 2. Parameters ---
MODEL_FILE = 'autoencoder_model.h5'
SCALER_FILE = 'scaler.joblib'
FINAL_THRESHOLD = 0.7931  # Your threshold

TEST_DURATION_SECONDS = 30
WINDOW_SECONDS = 5.0
FREQUENCY_HZ = 50
SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * FREQUENCY_HZ)
STEP_SIZE = 10
TREMOR_BAND = [4, 6]

# --- CHANGE 2: Add the feature names (MUST match your Colab script) ---
FEATURE_COLUMNS = [
    'ax_std', 'ay_std', 'az_std', 'gx_std', 'gy_std', 'gz_std',
    'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
    'ax_4_6hz_pwr', 'ay_4_6hz_pwr', 'az_4_6hz_pwr',
    'gx_4_6hz_pwr', 'gy_4_6hz_pwr', 'gz_4_6hz_pwr'
]

# --- 3. Socket Client Settings ---
PI_HOST_IP = '10.19.215.235'  # Your Pi's IP address
PORT = 9999 

# --- 4. Load Model and Scaler ---
try:
    # Use compile=False to fix the 'mae' loading error
    model = keras.models.load_model(MODEL_FILE, compile=False) 
    scaler = joblib.load(SCALER_FILE)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# --- 5. Global Variables ---
data_buffer = collections.deque(maxlen=SAMPLES_PER_WINDOW)
sample_count = 0
all_error_scores = []

# --- 6. Main Test Execution ---
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to Pi at {PI_HOST_IP}:{PORT}...")
        s.connect((PI_HOST_IP, PORT))
        print("Connection successful!")
        f = s.makefile()

        print("\n--- Starting 30 Second Anomaly Test ---")
        print("Please have the user rest their hand in position.")
        
        start_time = time.time()
        for line in f:
            if time.time() - start_time > TEST_DURATION_SECONDS:
                break 
            
            data = json.loads(line)
            row = [data['ax'], data['ay'], data['az'], data['gx'], data['gy'], data['gz']]
            data_buffer.append(row)
            sample_count += 1
            
            if len(data_buffer) == SAMPLES_PER_WINDOW and (sample_count % STEP_SIZE == 0):
                window = np.array(data_buffer)
                
                std_devs = np.std(window, axis=0)
                means = np.mean(window, axis=0)
                powers_4_6hz = [calculate_band_power(window[:, i], FREQUENCY_HZ, TREMOR_BAND) for i in range(6)]
                feature_vector = np.concatenate((std_devs, means, np.array(powers_4_6hz)))
                
                # --- Anomaly Detection (This whole block is updated) ---
                
                # 1. Convert numpy array to a DataFrame with column names
                feature_df = pd.DataFrame(feature_vector.reshape(1, -1), columns=FEATURE_COLUMNS)

                # 2. Scale the features using the DataFrame
                scaled_features = scaler.transform(feature_df)
                
                # 3. Get model's reconstruction
                reconstruction = model.predict(scaled_features, verbose=0)
                
                # 4. Calculate the error score
                error = np.mean(np.abs(scaled_features - reconstruction))
                
                all_error_scores.append(error)
                print(f"Window {len(all_error_scores)} processed, error: {error:.4f}")

    print("\n--- Test Complete. Analyzing Results... ---")

    # --- 7. The Final Verdict ---
    if not all_error_scores:
        print("No data was collected. Please check the Pi publisher.")
    else:
        final_average_score = np.mean(all_error_scores)
        
        print(f"Total windows analyzed: {len(all_error_scores)}")
        print(f"Your Threshold: {FINAL_THRESHOLD:.4f}")
        print(f"Final Average Score: {final_average_score:.4f}")

        if final_average_score > FINAL_THRESHOLD:
            print("\nFinal Verdict: ANOMALY DETECTED")
        else:
            print("\nFinal Verdict: Movement Normal")

except ConnectionRefusedError:
    print(f"\n[Error] Connection refused. Is the publisher.py script running on the Pi?")
except KeyboardInterrupt:
    print("\nTest stopped by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
