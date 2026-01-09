


# --- Mode selection ---
mode = int(input("Select mode:\n1 = Train and Predict\n2 = Load model and Predict\nEnter mode (1 or 2): "))

# =============================================================================
# Common parameter
look_back = 25
# look_back = 3
# =============================================================================


if mode == 1:
    
    import os
    import numpy as np
    import pandas as pd
    import keras
    import matplotlib.pyplot as plt
    import joblib
    import random
    import tensorflow as tf
    import time
    import csv
    from pathlib import Path
    try:
        import psutil
    except ImportError:
        psutil = None
        
    # from keras import Input
    from datetime import datetime
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout
    from keras.callbacks import EarlyStopping
    
    from sklearn.preprocessing import MinMaxScaler #, LabelEncoder
    # from sklearn.model_selection import train_test_split
        
    # ========================== LOAD FILES ==========================
    
    #folder_path = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Field data\\Coding_data"
    folder_path = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Field data\\Training_data\\Simple"
    
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    print(f"Found {len(excel_files)} Excel files in: {folder_path}")
    
    dataframes = []
    
    for i, file in enumerate(excel_files, 1):
        print(f"Reading file {i}/{len(excel_files)}: {file}")
        file_path = os.path.join(folder_path, file)
        df_i = pd.read_excel(file_path, sheet_name='data', skiprows=6, header=0,
                             usecols=['Time', 'Outlet Temp', 'Inlet Temp'])
        dataframes.append(df_i)


    if not dataframes:
        raise FileNotFoundError("No .xlsx files found or no data read from files.")

    df = pd.concat(dataframes, ignore_index=True)   # ✅ after the loop

       
    # --- Normalize the data ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Time', 'Outlet Temp', 'Inlet Temp']])
    scaled_df = pd.DataFrame(scaled, columns=['Time', 'Outlet Temp', 'Inlet Temp'])
    
    # --- Save the scaler ---
    
    scalers_path = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Scalers\\Simple"
    os.makedirs(scalers_path, exist_ok=True)
    
    scaler_file = os.path.join(scalers_path, f"scaler_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    
    # --- Prepare sequence data ---
    def create_dataset(dataset, look_back):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i+look_back, 0].reshape(-1, 1))  # Only Time input
            y.append(dataset[i+look_back, 1:])  # Target: Outlet + Inlet Temps
        return np.array(X), np.array(y)
    
    

    X, y = create_dataset(scaled, look_back)
    
    # --- Build the LSTM model ---
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    dropout_rate = 0.2   # 20% (standard in LSTM literature)
    LSTM_units=64
    
    
    
    model = Sequential([
     
    LSTM(LSTM_units, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(dropout_rate),
    

    LSTM(LSTM_units),
    Dropout(dropout_rate),

        Dense(2)  # Final output layer: [Outlet Temp, Inlet Temp]
    ])
   
    
    # 1 layer
    # model = Sequential([
    #     LSTM(LSTM_units, input_shape=(look_back, 1)),  # return_sequences=False (default)
    #     Dropout(dropout_rate),
    #     Dense(2)
    #     ])

    
    # model.compile(optimizer='adam', loss='mse')
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    
    early_stop = EarlyStopping(
    monitor='val_loss',       # or 'loss' if no validation split
    patience=15,              # stops after 15 epochs of no improvement
    restore_best_weights=True
    )
     
    #=========================================================================================
    
    log_dir = r"C:\Users\chanc\Desktop\Ph.D\20250131_LSTM Energy piles_GEOMATE2025\Training_logs\Simple"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    def unique_path(folder, base, ext=".csv"):
        p = Path(folder) / f"{base}{ext}"
        if not p.exists():
            return str(p)
        k = 2
        while True:
            p2 = Path(folder) / f"{base}_v{k}{ext}"
            if not p2.exists():
                return str(p2)
            k += 1
    
    def get_process_ram_mb():
        if psutil is None:
            return None
        proc = psutil.Process()
        return proc.memory_info().rss / (1024**2)
    
    def get_gpu_mem_mb():
        # Best-effort; may fail depending on TF version/device
        try:
            info = tf.config.experimental.get_memory_info('GPU:0')
            # returns bytes in TF; convert to MB
            cur = info.get('current', None)
            peak = info.get('peak', None)
            cur_mb = cur / (1024**2) if cur is not None else None
            peak_mb = peak / (1024**2) if peak is not None else None
            return cur_mb, peak_mb
        except Exception:
            return None, None

    #======================================================================================
    
    # --- Train the model ---
    start_time = time.perf_counter()
    
    history = model.fit(
        X, y,
        epochs=200,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )
    
    train_seconds = time.perf_counter() - start_time
    
    
    #======================================================================================
    
    # ---- derive "best epoch" + best metrics ----
    best_epoch = None
    best_val_loss = None
    best_val_mse = None
    
    if 'val_loss' in history.history:
        best_epoch = int(np.argmin(history.history['val_loss'])) + 1
        best_val_loss = float(history.history['val_loss'][best_epoch - 1])
    
    if 'val_mse' in history.history:
        best_val_mse = float(history.history['val_mse'][best_epoch - 1]) if best_epoch else float(np.min(history.history['val_mse']))
    
    # ---- you set this manually each run ----
    num_lstm_layers = int(input("Enter number of LSTM layers used in this run (1-5): "))
    
    # ---- cost proxies ----
    params = int(model.count_params())
    ram_mb = get_process_ram_mb()
    gpu_cur_mb, gpu_peak_mb = get_gpu_mem_mb()
    
    # ---- append to single benchmark file (no duplicates) ----
    benchmark_path = os.path.join(log_dir, "architecture_benchmark.csv")
    file_exists = os.path.exists(benchmark_path)
    
    row = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "look_back": look_back,
        "num_lstm_layers": num_lstm_layers,
        "lstm_units": LSTM_units,
        "dropout_rate": dropout_rate,
        "epochs_max": 200,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mse": best_val_mse,
        "train_seconds": train_seconds,
        "params": params,
        "ram_mb": ram_mb,
        "gpu_current_mb": gpu_cur_mb,
        "gpu_peak_mb": gpu_peak_mb,
    }
    
    cols = list(row.keys())
    
    with open(benchmark_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    
    print(f"Benchmark row appended to: {benchmark_path}")
    
    #======================================================================================
  
    
    # ======================== SAVE MODEL ============================
    
    model_save_path = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Trained_model\\Simple"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Function to generate a unique filename with timestamp
    def get_timestamped_filename(directory, base_name, extension=".keras"):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"{base_name}_{timestamp}{extension}"
        return os.path.join(directory, filename)
    
    # Get a timestamped model path
    model_file_path = get_timestamped_filename(model_save_path, "trained_model")
    
    # Save the model
    keras.saving.save_model(model, model_file_path)
    print(f"Model saved to {model_file_path}")
    # ================================================================
    
    
    # ======================== Save history to CSV and show best epoch ============================
    hist_df = pd.DataFrame(history.history)
    hist_save_dir = r"C:\Users\chanc\Desktop\Ph.D\20250131_LSTM Energy piles_GEOMATE2025\Training_logs\Simple"
    os.makedirs(hist_save_dir, exist_ok=True)
    
    hist_path = unique_path(hist_save_dir, f"history_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    hist_df.to_csv(hist_path, index=False)
    print(f"Training history saved to {hist_path}")

    
    if 'val_loss' in history.history:
        best_epoch = int(np.argmin(history.history['val_loss'])) + 1
        print(f"Best epoch (min val_loss): {best_epoch} "
          f"val_loss={history.history['val_loss'][best_epoch-1]:.6f}")   
    # =============================================================================================
    
    
    # --- User input for prediction interval and points ---
    prediction_interval = float(input("Enter the time step between predictions (e.g., 1/24 for every hour): "))
    prediction_points = int(input("Enter the number of future points you want to predict: "))
    
    # --- Predict future values ---
    last_seq = scaled[-look_back:, 0].reshape(1, look_back, 1)
    future_preds = []
    future_times = []
    
    next_time = scaled[-1, 0]  # in scaled time
    step = prediction_interval / (df['Time'].max() - df['Time'].min())  # normalize step
    
    for _ in range(prediction_points):
        pred = model.predict(last_seq, verbose=0)
        future_preds.append(pred[0])
        next_time += step
        future_times.append(next_time)
        last_seq = np.append(last_seq[:, 1:, :], [[[next_time]]], axis=1)
    
    # --- Convert predictions back to original scale ---
    future_scaled = np.column_stack((future_times, np.array(future_preds)))
    future_unscaled = scaler.inverse_transform(future_scaled)
    
    pred_df = pd.DataFrame(future_unscaled, columns=['Time', 'Outlet Temp', 'Inlet Temp'])
    print("\nPredicted Future Temperatures:")
    print(pred_df)
    
    
    # ======================== Save predictions to Excel file ============================
    prediction_save_path = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Predicted_Results\\Simple"
    os.makedirs(prediction_save_path, exist_ok=True)
    
    # Create a timestamped filename
    excel_filename = f"predicted_temperatures_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.xlsx"
    excel_filepath = os.path.join(prediction_save_path, excel_filename)
    
    # Save DataFrame to Excel
    pred_df.to_excel(excel_filepath, index=False)
    print(f"Predicted results saved to {excel_filepath}")
    # ====================================================================================
    
    
    # --- Plot the results ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Outlet Temp'], label='Actual Outlet Temp')
    plt.plot(df['Time'], df['Inlet Temp'], label='Actual Inlet Temp')
    plt.plot(pred_df['Time'], pred_df['Outlet Temp'], 'o--', label='Predicted Outlet Temp')
    plt.plot(pred_df['Time'], pred_df['Inlet Temp'], 'o--', label='Predicted Inlet Temp')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature Prediction with LSTM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
elif mode == 2:
    print("\nhello\n")
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    
    from keras.models import load_model
    from datetime import datetime
   
    
    # =============================Load trained model =======================================  
    model_folder = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Trained_model\\Simple"
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError("No .keras model files found.")

    print("Available models:")
    for i, f in enumerate(model_files):
        print(f"{i + 1}: {f}")
    choice = int(input("Select model file number: ")) - 1
    model_path = os.path.join(model_folder, model_files[choice])
    model = load_model(model_path)
    print(f"Loaded model: {model_files[choice]}")
    # =======================================================================================

    
    # =============================Load full scaler =========================================
    scaler_folder = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Scalers\\Simple"
    
    # Find all .pkl files regardless of naming
    all_scaler_files = [f for f in os.listdir(scaler_folder) if f.endswith('.pkl')]
    
    if not all_scaler_files:
        raise FileNotFoundError("No .pkl files found in the Scalers folder.")
    
    print("Available scaler files:")
    for idx, fname in enumerate(all_scaler_files):
        print(f"{idx + 1}. {fname}")
    
    # Prompt user to choose a file
    selected_index = input(f"Select a scaler file to load [1-{len(all_scaler_files)}]: ")
    
    # Validate input
    try:
        selected_index = int(selected_index)
        if not (1 <= selected_index <= len(all_scaler_files)):
            raise ValueError
    except ValueError:
        raise ValueError("Invalid selection. Please enter a valid number from the list.")
    
    # Load selected scaler
    selected_file = all_scaler_files[selected_index - 1]
    all_feature_scalers_path = os.path.join(scaler_folder, selected_file)
    
    print(f"Loading scaler from: {all_feature_scalers_path}")
    scalers = joblib.load(all_feature_scalers_path)

    # =======================================================================================
    
    # ============================= Load preprocessed data ==================================   
    preprocess_folder = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Preprocess\\Simple"
    
    def select_preprocess_file(folder):
        valid_exts = ('.csv', '.xls', '.xlsx')
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

        if not files:
            raise FileNotFoundError(f"No preprocessed data files with extensions {valid_exts} found in {folder}")

        print("\nAvailable Preprocessed Data Files:")
        for idx, fname in enumerate(files):
            print(f"{idx + 1}. {fname}")

        selected_index = input(f"Select a file to load [1-{len(files)}]: ")

        try:
            selected_index = int(selected_index)
            if not (1 <= selected_index <= len(files)):
                raise ValueError
        except ValueError:
            raise ValueError("Invalid selection for preprocess file.")

        selected_file = files[selected_index - 1]
        print(f"Loading data from: {selected_file}")
        return os.path.join(folder, selected_file)

          
    preprocess_file = select_preprocess_file(preprocess_folder)

    # --- Load selected preprocessed file into DataFrame ---
    if preprocess_file.endswith('.csv'):
        final_df = pd.read_csv(preprocess_file)
    elif preprocess_file.endswith(('.xls', '.xlsx')):
        final_df = pd.read_excel(preprocess_file, sheet_name=0)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    # --- Clean column names ---
    final_df.columns = [col.strip().strip("'").strip('"') for col in final_df.columns]

    # --- Check for required columns ---
    required_cols = [
        'Time', 'Inlet Temp', 'Outlet Temp'
    ]
    missing = [col for col in required_cols if col not in final_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(preprocess_file)}: {missing}")

    # --- Filter and clean ---
    final_df = final_df[required_cols]
    final_df['Time'] = pd.to_numeric(final_df['Time'], errors='coerce')
    final_df.dropna(subset=['Time'], inplace=True)

    # --- Check if enough sequence data ---
    if len(final_df) < look_back:
        raise ValueError(f"Not enough data: at least {look_back} rows required.")
    # =======================================================================================  
    
    # --- Normalize the input ---
    scaled = scalers.transform(final_df[['Time', 'Outlet Temp', 'Inlet Temp']])
    print("\nNormalize the input\n")
    
       # --- Prepare the last input sequence ---
    last_seq = scaled[-look_back:, 0].reshape(1, look_back, 1)
    
    # --- User input for prediction interval and steps ---
    prediction_interval = float(input("Enter the time step between predictions (e.g., 1/24 for every hour): "))
    prediction_points = int(input("Enter the number of future points to predict: "))
    
    # --- Predict multiple steps into the future ---
    next_time = scaled[-1, 0]
    step = prediction_interval / (final_df['Time'].max() - final_df['Time'].min())  # normalized step
    future_preds = []
    future_times = []
    
    for _ in range(prediction_points):
        pred = model.predict(last_seq, verbose=0)
        future_preds.append(pred[0])
        next_time += step
        future_times.append(next_time)
        last_seq = np.append(last_seq[:, 1:, :], [[[next_time]]], axis=1)
    
    # --- Inverse transform to original scale ---
    future_scaled = np.column_stack((future_times, np.array(future_preds)))
    future_unscaled = scalers.inverse_transform(future_scaled)
    
    pred_df = pd.DataFrame(future_unscaled, columns=['Time', 'Outlet Temp', 'Inlet Temp'])
    
    # --- Save to Excel ---
    output_folder = "C:\\Users\\chanc\\Desktop\\Ph.D\\20250131_LSTM Energy piles_GEOMATE2025\\Predicted_Results\\Simple"
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    save_path = os.path.join(output_folder, f'prediction_temperatures_{timestamp}.xlsx')
    pred_df.to_excel(save_path, index=False)
    print(f"Predicted results saved to {save_path}")
    
    # --- Print predictions ---
    print("\nPredicted Future Temperatures:")
    print(pred_df)
    
    # --- Combine and plot actual + predicted results ---
    plt.figure(figsize=(10, 5))
    
    # Plot actual data from preprocessed file
    plt.plot(final_df['Time'], final_df['Inlet Temp'], label='Actual Inlet Temp')
    plt.plot(final_df['Time'], final_df['Outlet Temp'], label='Actual Outlet Temp')
    
    
    # Plot predicted future values
    plt.plot(pred_df['Time'], pred_df['Inlet Temp'], 'o--', label='Predicted Inlet Temp')
    plt.plot(pred_df['Time'], pred_df['Outlet Temp'], 'o--', label='Predicted Outlet Temp')
    
    
    # Plot formatting
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature Prediction (Extended)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
    