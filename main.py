import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception as e:
    print(f"Error: {e}")
    model_loaded = False

try:
    df = pd.read_csv("./it/it.csv").dropna().drop_duplicates()
    df["date"] = pd.to_datetime(df["date"])
    print("CSV loaded successfully with", len(df), "entries")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load CSV: {e}")
    df = pd.DataFrame(columns=["Date", "Close"])

def predict_close():
    try:
        if not model_loaded:
            messagebox.showerror("Error", "Model is not loaded. Please check model.pkl")
            return

        # Get user inputs
        date = date_entry.get()
        open_price = float(open_entry.get())
        high_price = float(high_entry.get())
        low_price = float(low_entry.get())

        # Convert Date to datetime
        date_obj = pd.to_datetime(date)
        year, month, day = date_obj.year, date_obj.month, date_obj.day

        # Prepare input for the model
        features = pd.DataFrame([[open_price, high_price, low_price, year, month, day]],
                                columns=["open", "high", "low", "year", "month", "day"])

        # Make prediction
        predicted_close = model.predict(features)[0]
        result_label.config(text=f"Predicted Close Price: {predicted_close:.2f}")
        update_graph(date_obj, predicted_close)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def update_graph(date, predicted_close):
    if df.empty:
        messagebox.showerror("Error", "No data available to display actual close prices.")
        return

    # Clear previous graph
    ax.clear()
    
    # Plot actual close prices
    ax.plot(df["date"], df["close"], label="Actual Close Price", color="blue")
    
    # Plot predicted close price as a single point
    ax.scatter([date], [predicted_close], color="red", label="New Prediction", zorder=3)
    
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Close Price", fontsize=14)
    ax.set_title("Actual Close Price Over Time", fontsize=16)
    ax.legend()
    ax.grid()
    fig.autofmt_xdate()
    canvas.draw()

# Create main window
root = tk.Tk()
root.title("Stock Price Predictor")
root.geometry("1200x800")

# Labels and Entry Fields
tk.Label(root, text="Enter Date (DD-MM-YYYY):", font=("Arial", 14)).pack()
date_entry = tk.Entry(root, font=("Arial", 14))
date_entry.pack()

tk.Label(root, text="Enter Open Price:", font=("Arial", 14)).pack()
open_entry = tk.Entry(root, font=("Arial", 14))
open_entry.pack()

tk.Label(root, text="Enter High Price:", font=("Arial", 14)).pack()
high_entry = tk.Entry(root, font=("Arial", 14))
high_entry.pack()

tk.Label(root, text="Enter Low Price:", font=("Arial", 14)).pack()
low_entry = tk.Entry(root, font=("Arial", 14))
low_entry.pack()

# Predict Button
predict_btn = ttk.Button(root, text="Predict Close Price", command=predict_close)
predict_btn.pack(pady=10)
predict_btn.config(width=20)
if not model_loaded:
    predict_btn.config(state=tk.DISABLED)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
result_label.pack()

# Matplotlib Figure
fig, ax = plt.subplots(figsize=(12, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Run Tkinter Event Loop
root.mainloop()
