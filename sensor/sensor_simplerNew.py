
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
COMPORT = "COM10"
SAMPLES=200

# print([port.device for port in serial.tools.list_ports.comports()])
arduino = serial.Serial()

class App:
    
    # def forward(self, x):
    #     A = 43.20978
    #     B = 0.286
    #     C = 1.0373300741204
    #     K = 900.0010210040905
    #     v = 0.1
    #     real = A + (K - A)/ (1 + np.exp(-1 * B(x-C)))**(1/v)
    
    def inverse(self, y): 
        # A = 43.20978
        # B = 0.286
        # C = 1.0373300741204
        # K = 900.0010210040905
        # v = 0.1
        A  = -9.824360913609562
        K  = 871.4744633266955
        B  = 0.2909366235093304
        C  = 4.3307594408159495
        v = 0.2822807132259202
        y = float(y)
        #print(y)
        real = C - (1.0 / B) * np.log((( (K - A) / (y - A) ) ** v) - 1.0)
        return real
    
    def __init__(self):
        self.running = True
        self.paused = False
        self.snap = False
        self.counter = 0
        self.dist = False

        self.current_dist = 1
        self.dataset = {}
        self.trialNumber = 0

        self.arduino = serial.Serial(port=COMPORT, baudrate=115200, timeout=5)

    def run_machine(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            while self.arduino.is_open and self.running:
                try:
                    data = self.arduino.readline().decode(errors='ignore')
                except:
                    print("Triggered Termination")
                    break

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
                    self.current_dist = float(msg)
                    self.dataset[self.current_dist] = []
                    junk = self.arduino.read_all()
                    self.dist = False

                else:
                    try:
                        data = self.arduino.readline().decode(errors='ignore')
                        if self.snap and self.counter < SAMPLES:
                            if (self.counter == 0):
                                self.dataset[self.current_dist] = []
                            real = data.strip()
                            #real = self.inverse(real)
                            # print( self.inverse(real) )
                            print(real)
                            self.dataset[self.current_dist].append(real)
                            self.counter += 1
                            # if (self.counter == SAMPLES):
                            #     temp = 0
                            #     for value in self.dataset[self.current_dist] :
                            #         temp += float(value)
                            #     temp = temp / 99
                            #     print("Average distance guessed", self.inverse(temp))

                        else:
                            self.snap = False
                            self.counter = 0
                        
                    except KeyboardInterrupt:
                        print("Triggered Termination")
                        break

            print("Arduino Disconnected!") 
            print("====")
            print(self.dataset)
            df = pd.DataFrame(self.dataset)
            #lb is taped
            df.to_csv(r'C:\Users\k28ad\OneDrive\Documents\sensor\dataSensor3New.csv')

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