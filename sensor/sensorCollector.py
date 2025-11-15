
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
COMPORT = "COM9"
SAMPLES = 100
SENSOR = 2 #0 is taped sensor, 1 is non-taped sensor. s

# print([port.device for port in serial.tools.list_ports.comports()])
arduino = serial.Serial()

class App:
    
    def map(self, analog, sensor): 

        #0 is taped, 1 is nontaped
        A  = -8.976076325826309
        K  = 913.5463710698101
        B  = 0.29767471011439534
        C  = 5.6686184386250025
        v = 0.3627635461289861  

        if (sensor == 1):
            A  = -4.831976283950702
            K  = 885.9877001844566
            B  = 0.2793284618109283
            C  = 3.8852507844119217
            v = 0.2389935455347361
        
        if (sensor == 2):
            A  = -9.824360913609562
            K  = 871.4744633266955
            B  = 0.2909366235093304
            C  = 4.3307594408159495
            v = 0.2822807132259202

        if (sensor == 3):
            A  = -13.891292062248292
            K  = 990.6819962477331
            B  = 0.16376045588859353
            C  = -0.074904004740735
            v = 0.17727132893449118

        #IMPORTANT: FOR SENSOR 3, INDUCTION SENSOR OUTPUTS VOLTAGE > 6 v
        #SO WE USED NEW VOLTAGE DIVIDER: 220 OHMS, 550 OHMS
        y = float(analog)
        #print(y)
        real = C - (1.0 / B) * np.log((( (K - A) / (y - A) ) ** v) - 1.0)
        return real
    
    def __init__(self):
        self.running = True
        self.paused = True
        self.collecting = False
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
                    
                    sleep(0.1)
                    msg = input("type 'c' to show data, type 'p' to pause, type 'esc' to pause: ").strip()
                    self.arduino.write(bytes(msg, 'utf-8')) 
                    self.collecting = False
                    if (msg == 'c'):
                        self.collecting = True
                        self.paused = False
                    elif (msg == 'q'):
                        self.running = False
                    sleep(0.5)

                elif self.collecting:
                    try:
                        data = self.arduino.readline().decode(errors='ignore')
                        data = data.strip()
                        print(data)
                        distance = self.map(data, SENSOR)
                        print(distance, "mm")
                        # distance holds sensor values

                    except KeyboardInterrupt:
                        print("Triggered Termination")
                        break

            listener.stop()
            arduino.close()
            return
        
    def on_press(self, key):
        try:
            if key.char == 'p':
                self.paused = True
                
        except:
            if key == keyboard.Key.esc:
                print("pausing program")
                self.pause = True

app = App()
app.run_machine()