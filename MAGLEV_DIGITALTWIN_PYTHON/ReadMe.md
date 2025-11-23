# How To Use
## Running the Simulation
Run ```pip install -r requirements.txt``` followed by the desired simulation file. Or, if your environment is already set up, just run the python file.
### Single Simulation
Run ```SimulateSingle.py```. You must exit the visualization window in order to see the data plots. <br> Generated files will be saved to ```sim_results/``` in the directory where the python script is run from (likely the root directory of your repository clone).
### Multiple Simulations with parameter noise
Set desired parameter noise level in ```SimulateMultipleWithNoise.py```, then run. Noise is applied to electromagnetic characteristics, length, width, sensor position, yoke position, moment of inertia, and mass. <br> Generated files will be saved to ```sim_results_multi/```.
## Modifying the PID control algorithm
Modify ```controller.py```. You will see constants related to heave, pitch, and roll controllers. Will update to include current control and will make simulation much more accurate soon.
## Modifying pod parameters
```parameters.py``` provides all necessary mechanical parameters. Yoke and sensor locations are matrices formed by putting together 4 column vectors of the form $\begin{bmatrix}x\\y\\z\end{bmatrix}$ relative to the center of mass. <br><br>```fmag2()``` within ```utils.py``` deals with the magnetic parameters.
