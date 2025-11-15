
import serial
import serial.tools.list_ports
import pandas as pd
import os
from pynput import keyboard
from datetime import datetime
from time import sleep
import sys
import numpy as np

#import termios
import tkinter as tk


# COMPORT = "/dev/cu.usbmodem11301"
COMPORT = "COM5"
SAMPLES=100
# a = 827
# b = .524
# c = 8.59834

# print([port.device for port in serial.tools.list_ports.comports()])
arduino = serial.Serial()

class App:
    
    
    def __init__(self):
        self.running = True
        self.paused = False
        self.snap = False
        self.counter = 0
        self.dist = False

        self.current_dist = 1

        self.dataset = {
            i: [0 for _ in range(SAMPLES)] for i in range(1, 25)
        }

        self.arduino = serial.Serial(port=COMPORT, baudrate=115200, timeout=5)

    def run_machine(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            while self.arduino.is_open and self.running:
                if self.paused:
                    ## PILOT--SENDING INPUTS, SWITCHING SENSOR INDEX
                    sleep(0.1)
                    #termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    msg = input("Enter message: ").strip()
                    self.arduino.write(bytes(msg, 'utf-8')) 
                    self.paused = False
                    sleep(0.5)

                elif self.dist:
                    sleep(0.1)
                    #termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    msg = input("Enter dist: ")
                    self.current_dist = int(msg)
                    junk = self.arduino.read_all()
                    self.dist = False

                else:
                    try:
                        data = self.arduino.readline().decode(errors='ignore')
                        if self.snap and self.counter < SAMPLES:
                            real = data.strip()
                            print(real)
                            self.dataset[self.current_dist][self.counter] = int(real)

                            # mm = 100000
                            # if (int(real) != 0):
                            #     mm = c - (1/b) * np.log(np.abs(a / int(real) - 1))
                            # print(mm,"mm")

                            # self.dataset_mm[self.current_dist][self.counter] = mm
                            self.counter += 1
                        else:
                            self.snap = False
                            self.counter = 0
                        
                    except KeyboardInterrupt:
                        print("Triggered Termination")
                        break

            print("Arduino Disconnected!")  
            for value in self.dataset[1]:
                print(value)

            print("====")
            df = pd.DataFrame(self.dataset)
            
            df.to_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\dataRB.csv')
            # df2 = pd.DataFrame(self.dataset_mm)
            # df2.to_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\dataLBMM.csv')

            listener.stop()
            arduino.close()
            return
        
    def on_press(self, key):
        try:
            if key.char == 'p':
                self.paused = True

            elif key.char == 's':
                self.snap = True
            
            elif key.char == 'd':
                self.dist = True
                # def submit_slider1():
                #     value = slider1.get()
                #     self.current_dist = value

                # def update_label1(val):
                #     label1.config(text=f"Value: {val}")

                # # Create main window
                # root = tk.Tk()
                # root.title("Slider Example")
                # root.geometry("300x200")

                # # Slider 1
                # slider1 = tk.Scale(root, from_=0, to=3, orient=tk.HORIZONTAL, command=update_label1)
                # slider1.pack()
                # label1 = tk.Label(root, text="Value: 0")
                # label1.pack()
                # button1 = tk.Button(root, text="Submit Slider 1", command=submit_slider1)
                # button1.pack()

                # root.mainloop()
                
        except:
            if key == keyboard.Key.esc:
                print("are you here??")
                self.running = False



app = App()
app.run_machine()
