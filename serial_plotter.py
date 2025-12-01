import serial
import serial.tools.list_ports
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend explicitly
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import argparse
import time
import random
import threading
import sys
from datetime import datetime
import queue
import tkinter as tk
from tkinter import ttk

# Constants
BAUD_RATE = 2000000
MAX_POINTS = 100  # Number of points to display on the plot

# Save queue for handling saves in main thread
save_queue = queue.Queue()

# Data storage
data = {
    "Left": deque(maxlen=MAX_POINTS),
    "Right": deque(maxlen=MAX_POINTS),
    "Front": deque(maxlen=MAX_POINTS),
    "Back": deque(maxlen=MAX_POINTS),
    "Avg": deque(maxlen=MAX_POINTS),
    "FLPWM": deque(maxlen=MAX_POINTS),
    "BLPWM": deque(maxlen=MAX_POINTS),
    "FRPWM": deque(maxlen=MAX_POINTS),
    "BRPWM": deque(maxlen=MAX_POINTS),
    "ControlOn": deque(maxlen=MAX_POINTS),
    "Time": deque(maxlen=MAX_POINTS)
}

# Recording data storage (no max length limit)
recording_data = {
    "Left": [],
    "Right": [],
    "Front": [],
    "Back": [],
    "Avg": [],
    "FLPWM": [],
    "BLPWM": [],
    "FRPWM": [],
    "BRPWM": [],
    "ControlOn": [],
    "Time": []
}

start_time = time.time()
recording = False
recording_start_time = None

def get_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"{i}: {p.device} - {p.description}")
    
    if len(ports) == 1:
        return ports[0].device
    
    try:
        selection = int(input("Select port index: "))
        return ports[selection].device
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None

def mock_data_generator():
    """Generates mock data in the expected CSV format."""
    while True:
        # Simulate sensor readings (around 12.0mm with noise)
        left = 12.0 + random.uniform(-0.5, 0.5)
        right = 12.0 + random.uniform(-0.5, 0.5)
        front = 12.0 + random.uniform(-0.5, 0.5)
        back = 12.0 + random.uniform(-0.5, 0.5)
        avg = (left + right + front + back) / 4.0
        
        # Simulate PWM values (around 0 with noise)
        fl_pwm = random.randint(-50, 50)
        bl_pwm = random.randint(-50, 50)
        fr_pwm = random.randint(-50, 50)
        br_pwm = random.randint(-50, 50)
        
        control_on = 1
        
        # CSV Format: Left,Right,Front,Back,Avg,FLPWM,BLPWM,FRPWM,BRPWM,ControlOn
        line = f"{left:.2f},{right:.2f},{front:.2f},{back:.2f},{avg:.2f},{fl_pwm},{bl_pwm},{fr_pwm},{br_pwm},{control_on}"
        yield line
        time.sleep(0.01) # 100Hz

def read_serial(ser, mock=False):
    """Reads data from serial port or mock generator."""
    global recording, recording_start_time
    generator = mock_data_generator() if mock else None
    
    while True:
        try:
            if mock:
                line = next(generator)
            else:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                else:
                    continue
            
            parts = line.split(',')
            if len(parts) == 10:
                current_time = time.time() - start_time
                
                left = float(parts[0])
                right = float(parts[1])
                front = float(parts[2])
                back = float(parts[3])
                avg = float(parts[4])
                flpwm = float(parts[5])
                blpwm = float(parts[6])
                frpwm = float(parts[7])
                brpwm = float(parts[8])
                control_on = int(parts[9])
                
                # Update live display data
                data["Left"].append(left)
                data["Right"].append(right)
                data["Front"].append(front)
                data["Back"].append(back)
                data["Avg"].append(avg)
                data["FLPWM"].append(flpwm)
                data["BLPWM"].append(blpwm)
                data["FRPWM"].append(frpwm)
                data["BRPWM"].append(brpwm)
                data["ControlOn"].append(control_on)
                data["Time"].append(current_time)
                
                # If recording, store in unlimited buffer
                if recording:
                    recording_time = current_time - recording_start_time
                    recording_data["Left"].append(left)
                    recording_data["Right"].append(right)
                    recording_data["Front"].append(front)
                    recording_data["Back"].append(back)
                    recording_data["Avg"].append(avg)
                    recording_data["FLPWM"].append(flpwm)
                    recording_data["BLPWM"].append(blpwm)
                    recording_data["FRPWM"].append(frpwm)
                    recording_data["BRPWM"].append(brpwm)
                    recording_data["ControlOn"].append(control_on)
                    recording_data["Time"].append(recording_time)
                
        except ValueError:
            pass # Ignore parse errors
        except Exception as e:
            print(f"Error reading data: {e}")
            if not mock:
                break

def save_recording(metadata=None):
    """Save the recorded data to a high-quality PNG file."""
    global recording_data
    
    if not recording_data["Time"]:
        print("No data recorded to save.")
        return
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if it doesn't exist
    import os
    save_dir = "tuningTrials"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{save_dir}/trial_{timestamp}.png"
    
    # Create a completely independent figure with Agg backend
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    save_fig = Figure(figsize=(14, 10))
    canvas = FigureCanvasAgg(save_fig)
    save_fig.suptitle(f'Levitation Control Trial - {timestamp}', fontsize=16, fontweight='bold')
    
    # Plot Sensors
    ax1 = save_fig.add_subplot(2, 1, 1)
    ax1.plot(recording_data["Time"], recording_data["Left"], label="Left", linewidth=2)
    ax1.plot(recording_data["Time"], recording_data["Right"], label="Right", linewidth=2)
    ax1.plot(recording_data["Time"], recording_data["Front"], label="Front", linewidth=2)
    ax1.plot(recording_data["Time"], recording_data["Back"], label="Back", linewidth=2)
    ax1.plot(recording_data["Time"], recording_data["Avg"], label="Avg", linestyle='--', color='black', linewidth=2.5)
    ax1.set_ylabel("Distance (mm)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title("Sensor Readings", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, recording_data["Time"][-1]])
    
    # Plot PWMs
    ax2 = save_fig.add_subplot(2, 1, 2)
    ax2.plot(recording_data["Time"], recording_data["FLPWM"], label="FL", linewidth=2)
    ax2.plot(recording_data["Time"], recording_data["BLPWM"], label="BL", linewidth=2)
    ax2.plot(recording_data["Time"], recording_data["FRPWM"], label="FR", linewidth=2)
    ax2.plot(recording_data["Time"], recording_data["BRPWM"], label="BR", linewidth=2)
    ax2.set_ylabel("PWM Value", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title("PWM Outputs", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, recording_data["Time"][-1]])
    
    save_fig.tight_layout()
    
    # Save with high DPI using the canvas
    canvas.print_figure(filename, dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*60}")
    print(f"Trial data saved to: {filename}")
    print(f"Duration: {recording_data['Time'][-1]:.2f} seconds")
    print(f"Data points: {len(recording_data['Time'])}")
    
    # Save metadata if provided
    if metadata:
        txt_filename = f"{save_dir}/trial_{timestamp}.txt"
        try:
            with open(txt_filename, 'w') as f:
                f.write(f"Trial Timestamp: {timestamp}\n")
                f.write(f"Duration: {recording_data['Time'][-1]:.2f} seconds\n")
                f.write(f"Data Points: {len(recording_data['Time'])}\n\n")
                
                f.write("=== Reference Values ===\n")
                if "references" in metadata:
                    for key, value in metadata["references"].items():
                        f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("=== PID Parameters ===\n")
                if "pid" in metadata:
                    for mode, values in metadata["pid"].items():
                        f.write(f"[{mode}]\n")
                        f.write(f"  Kp: {values['Kp']}\n")
                        f.write(f"  Ki: {values['Ki']}\n")
                        f.write(f"  Kd: {values['Kd']}\n")
                f.write("\n")
            print(f"Metadata saved to: {txt_filename}")
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    print(f"{'='*60}\n")


def update_plot(frame):
    global recording
    
    # Check if there's a save request
    try:
        while not save_queue.empty():
            item = save_queue.get_nowait()
            if isinstance(item, dict):
                save_recording(metadata=item)
            else:
                save_recording()
    except:
        pass
    
    plt.clf()
    
    # Add recording indicator
    fig = plt.gcf()
    if recording:
        fig.patch.set_facecolor('#ffe6e6')  # Light red background when recording
    else:
        fig.patch.set_facecolor('white')
    
    # Plot Sensors
    plt.subplot(2, 1, 1)
    plt.plot(data["Time"], data["Left"], label="Left")
    plt.plot(data["Time"], data["Right"], label="Right")
    plt.plot(data["Time"], data["Front"], label="Front")
    plt.plot(data["Time"], data["Back"], label="Back")
    plt.plot(data["Time"], data["Avg"], label="Avg", linestyle='--', color='black')
    plt.ylabel("Distance (mm)")
    plt.legend(loc='upper right')
    title = "Sensor Readings"
    if recording:
        title += " [RECORDING]"
    plt.title(title)
    plt.grid(True)
    
    # Plot PWMs
    plt.subplot(2, 1, 2)
    plt.plot(data["Time"], data["FLPWM"], label="FL")
    plt.plot(data["Time"], data["BLPWM"], label="BL")
    plt.plot(data["Time"], data["FRPWM"], label="FR")
    plt.plot(data["Time"], data["BRPWM"], label="BR")
    plt.ylabel("PWM Value")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right')
    title = "PWM Outputs"
    if recording:
        title += " [RECORDING]"
    plt.title(title)
    plt.grid(True)
    
    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description='Serial Plotter for Levitation Control')
    parser.add_argument('--port', type=str, help='Serial port to connect to')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of serial')
    args = parser.parse_args()
    
    ser = None
    if not args.mock:
        port = args.port or get_serial_port()
        if not port:
            print("No serial port found or selected. Use --mock to test without hardware.")
            return
        
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=1)
            print(f"Connected to {port} at {BAUD_RATE} baud.")
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return

    # Start data reading thread
    thread = threading.Thread(target=read_serial, args=(ser, args.mock), daemon=True)
    thread.start()
    
    # Send initial start command
    if ser:
        try:
            print("Sending start command...")
            ser.write(b'1')
        except Exception as e:
            print(f"Failed to send start command: {e}")

    # Create Tkinter window
    root = tk.Tk()
    root.title("Levitation Control - Serial Plotter")
    
    # Create main container
    main_container = ttk.Frame(root)
    main_container.pack(fill=tk.BOTH, expand=True)
    
    # Left panel for controls
    control_panel = ttk.Frame(main_container, width=400)
    control_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
    
    # Right panel for plot
    plot_panel = ttk.Frame(main_container)
    plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # === CONTROL BUTTONS ===
    btn_frame = ttk.LabelFrame(control_panel, text="Control", padding=10)
    btn_frame.pack(fill=tk.X, pady=5)
    
    def start_control():
        if ser:
            ser.write(b'1')
            print("Control started")
            global recording, recording_start_time, recording_data
            recording = True
            recording_start_time = time.time() - start_time
            for key in recording_data:
                recording_data[key] = []
    
    def stop_control():
        if ser:
            ser.write(b'0')
            print("Control stopped")
            global recording
            if recording:
                recording = False
                
                # Collect metadata
                metadata = {
                    "references": {
                        "Avg Height": avg_ref_entry.get(),
                        "LR Diff": lr_diff_entry.get(),
                        "FB Diff": fb_diff_entry.get()
                    },
                    "pid": {}
                }
                
                # Collect PID values
                for name, _, _, _, _ in pid_modes:
                    kp_e, ki_e, kd_e = pid_entries[name]
                    metadata["pid"][name] = {
                        "Kp": kp_e.get(),
                        "Ki": ki_e.get(),
                        "Kd": kd_e.get()
                    }
                
                save_queue.put(metadata)
    
    ttk.Button(btn_frame, text="Start Control & Record", command=start_control, width=25).pack(pady=2)
    ttk.Button(btn_frame, text="Stop Control & Save", command=stop_control, width=25).pack(pady=2)
    
    # === REFERENCE VALUES ===
    ref_frame = ttk.LabelFrame(control_panel, text="Reference Values", padding=10)
    ref_frame.pack(fill=tk.X, pady=5)
    
    # Average Reference
    ttk.Label(ref_frame, text="Avg Height (mm):", width=15).grid(row=0, column=0, sticky=tk.W, pady=2)
    avg_ref_entry = ttk.Entry(ref_frame, width=12)
    avg_ref_entry.insert(0, "11.0")
    avg_ref_entry.grid(row=0, column=1, padx=5, pady=2)
    
    # Left-Right Diff Reference
    ttk.Label(ref_frame, text="LR Diff (mm):", width=15).grid(row=1, column=0, sticky=tk.W, pady=2)
    lr_diff_entry = ttk.Entry(ref_frame, width=12)
    lr_diff_entry.insert(0, "-2.0")
    lr_diff_entry.grid(row=1, column=1, padx=5, pady=2)
    
    # Front-Back Diff Reference
    ttk.Label(ref_frame, text="FB Diff (mm):", width=15).grid(row=2, column=0, sticky=tk.W, pady=2)
    fb_diff_entry = ttk.Entry(ref_frame, width=12)
    fb_diff_entry.insert(0, "0.0")
    fb_diff_entry.grid(row=2, column=1, padx=5, pady=2)
    
    def send_references():
        try:
            avg_ref = float(avg_ref_entry.get())
            lr_diff = float(lr_diff_entry.get())
            fb_diff = float(fb_diff_entry.get())
            cmd = f"REF,{avg_ref},{lr_diff},{fb_diff}\n"
            if ser:
                ser.write(cmd.encode('utf-8'))
                print(f"Sent References: Avg={avg_ref}, LR={lr_diff}, FB={fb_diff}")
            else:
                print(f"Mock mode - would send: {cmd.strip()}")
        except ValueError as e:
            print(f"Error: Invalid reference values - {e}")
    
    ttk.Button(ref_frame, text="Send References", command=send_references, width=25).grid(row=3, column=0, columnspan=2, pady=10)
    
    # === PID TUNING ===
    pid_frame = ttk.LabelFrame(control_panel, text="PID Tuning", padding=10)
    pid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Create scrollable frame
    canvas = tk.Canvas(pid_frame, height=500)
    scrollbar = ttk.Scrollbar(pid_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # PID modes with default values from the .ino file
    pid_modes = [
        ("Repelling", 0, 250, 0, 1000),
        ("Attracting", 1, 250, 0, 1000),
        ("RollLeftDown", 2, 0, 0, 100),
        ("RollLeftUp", 3, 0, 0, 100),
        ("RollFrontDown", 4, 0, 0, 500),
        ("RollFrontUp", 5, 0, 0, 500),
    ]
    
    pid_entries = {}
    
    for name, mode, default_kp, default_ki, default_kd in pid_modes:
        mode_frame = ttk.LabelFrame(scrollable_frame, text=name, padding=5)
        mode_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Kp
        ttk.Label(mode_frame, text="Kp:", width=5).grid(row=0, column=0, sticky=tk.W)
        kp_entry = ttk.Entry(mode_frame, width=12)
        kp_entry.insert(0, str(default_kp))
        kp_entry.grid(row=0, column=1, padx=5)
        
        # Ki
        ttk.Label(mode_frame, text="Ki:", width=5).grid(row=1, column=0, sticky=tk.W)
        ki_entry = ttk.Entry(mode_frame, width=12)
        ki_entry.insert(0, str(default_ki))
        ki_entry.grid(row=1, column=1, padx=5)
        
        # Kd
        ttk.Label(mode_frame, text="Kd:", width=5).grid(row=2, column=0, sticky=tk.W)
        kd_entry = ttk.Entry(mode_frame, width=12)
        kd_entry.insert(0, str(default_kd))
        kd_entry.grid(row=2, column=1, padx=5)
        
        # Send button
        def make_send_func(mode_num, kp_e, ki_e, kd_e, mode_name):
            def send_pid():
                try:
                    kp = float(kp_e.get())
                    ki = float(ki_e.get())
                    kd = float(kd_e.get())
                    cmd = f"PID,{mode_num},{kp},{ki},{kd}\n"
                    if ser:
                        ser.write(cmd.encode('utf-8'))
                        print(f"Sent {mode_name}: Kp={kp}, Ki={ki}, Kd={kd}")
                    else:
                        print(f"Mock mode - would send: {cmd.strip()}")
                except ValueError as e:
                    print(f"Error: Invalid PID values - {e}")
            return send_pid
        
        send_btn = ttk.Button(mode_frame, text="Send", 
                             command=make_send_func(mode, kp_entry, ki_entry, kd_entry, name),
                             width=10)
        send_btn.grid(row=3, column=0, columnspan=2, pady=5)
        
        pid_entries[name] = (kp_entry, ki_entry, kd_entry)
    
    # Send All button
    def send_all_pid():
        for name, mode, _, _, _ in pid_modes:
            kp_e, ki_e, kd_e = pid_entries[name]
            try:
                kp = float(kp_e.get())
                ki = float(ki_e.get())
                kd = float(kd_e.get())
                cmd = f"PID,{mode},{kp},{ki},{kd}\n"
                if ser:
                    ser.write(cmd.encode('utf-8'))
                    time.sleep(0.05)  # Small delay between commands
                print(f"Sent {name}: Kp={kp}, Ki={ki}, Kd={kd}")
            except ValueError as e:
                print(f"Error in {name}: {e}")
    
    send_all_frame = ttk.Frame(scrollable_frame)
    send_all_frame.pack(fill=tk.X, pady=10, padx=5)
    ttk.Button(send_all_frame, text="Send All PID Values", command=send_all_pid, width=25).pack()
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # === PLOT SETUP ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.tight_layout(pad=3.0)
    canvas_plot = FigureCanvasTkAgg(fig, master=plot_panel)
    canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Setup axes once
    ax1.set_ylabel("Distance (mm)")
    ax1.grid(True)
    ax1.set_title("Sensor Readings")
    
    ax2.set_ylabel("PWM Value")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)
    ax2.set_title("PWM Outputs")
    
    # Initialize line objects for efficient updating
    line_left, = ax1.plot([], [], label="Left")
    line_right, = ax1.plot([], [], label="Right")
    line_front, = ax1.plot([], [], label="Front")
    line_back, = ax1.plot([], [], label="Back")
    line_avg, = ax1.plot([], [], label="Avg", linestyle='--', color='black', linewidth=1.5)
    
    line_fl, = ax2.plot([], [], label="FL")
    line_bl, = ax2.plot([], [], label="BL")
    line_fr, = ax2.plot([], [], label="FR")
    line_br, = ax2.plot([], [], label="BR")
    
    # Create legends after line objects
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    # Track recording state to minimize updates
    _was_recording = False
    
    def update_plot(frame):
        global recording
        nonlocal _was_recording
        
        # Check if there's a save request
        try:
            while not save_queue.empty():
                item = save_queue.get_nowait()
                if isinstance(item, dict):
                    save_recording(metadata=item)
                else:
                    save_recording()
        except:
            pass
        
        # Only update background color when recording state changes
        if recording != _was_recording:
            if recording:
                fig.patch.set_facecolor('#ffe6e6')
                ax1.set_title("Sensor Readings [RECORDING]")
                ax2.set_title("PWM Outputs [RECORDING]")
            else:
                fig.patch.set_facecolor('white')
                ax1.set_title("Sensor Readings")
                ax2.set_title("PWM Outputs")
            _was_recording = recording
        
        # Update line data efficiently (no clear/replot)
        if len(data["Time"]) > 0:
            time_data = list(data["Time"])
            
            line_left.set_data(time_data, list(data["Left"]))
            line_right.set_data(time_data, list(data["Right"]))
            line_front.set_data(time_data, list(data["Front"]))
            line_back.set_data(time_data, list(data["Back"]))
            line_avg.set_data(time_data, list(data["Avg"]))
            
            line_fl.set_data(time_data, list(data["FLPWM"]))
            line_bl.set_data(time_data, list(data["BLPWM"]))
            line_fr.set_data(time_data, list(data["FRPWM"]))
            line_br.set_data(time_data, list(data["BRPWM"]))
            
            # Auto-scale axes
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
        
        canvas_plot.draw_idle()
    
    def on_close():
        print("Window closed. Exiting script.")
        if ser:
            ser.close()
        root.quit()
        root.destroy()
        sys.exit()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Start animation with slower interval for smoother GUI (100ms = 10 FPS is sufficient)
    ani = animation.FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False, blit=False)
    
    print("\n" + "="*60)
    print("SERIAL PLOTTER - GUI MODE")
    print("="*60)
    print("  Use buttons to start/stop control and recording")
    print("  Adjust PID values and click 'Send' to update Arduino")
    print("="*60 + "\n")
    
    root.mainloop()

if __name__ == "__main__":
    main()
