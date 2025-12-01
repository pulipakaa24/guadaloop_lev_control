# PID Tuning & Reference Control System - Implementation Guide

## Overview
This system allows real-time tuning of all 6 PID control modes and reference setpoints through a GUI interface. Changes are transmitted via serial communication and applied immediately to the controller.

## Control Modes

The system supports 6 distinct PID control modes:

1. **Repelling** (Mode 0) - Average height control when pushing away
2. **Attracting** (Mode 1) - Average height control when pulling in
3. **RollLeftDown** (Mode 2) - Left-right balance when left side needs to go down
4. **RollLeftUp** (Mode 3) - Left-right balance when left side needs to go up
5. **RollFrontDown** (Mode 4) - Front-back balance when front needs to go down
6. **RollFrontUp** (Mode 5) - Front-back balance when front needs to go up

## Reference Values

The system also supports updating three reference setpoints:

1. **Average Height Reference (avgRef)** - Target height for levitation (mm)
2. **Left-Right Difference Reference (LRDiffRef)** - Target balance between left and right sensors (mm)
   - Positive value: left side should be higher than right
   - Negative value: right side should be higher than left
3. **Front-Back Difference Reference (FBDiffRef)** - Target balance between front and back sensors (mm)
   - Positive value: front should be higher than back
   - Negative value: back should be higher than front

## Serial Protocol

### PID Command Format
```
PID,<mode>,<kp>,<ki>,<kd>\n
```

### Reference Command Format
```
REF,<avgRef>,<lrDiffRef>,<fbDiffRef>\n
```

### Parameters
**PID Command:**
- **mode**: Integer 0-5 (see control modes above)
- **kp**: Float - Proportional gain
- **ki**: Float - Integral gain
- **kd**: Float - Derivative gain

**Reference Command:**
- **avgRef**: Float - Average height target in mm
- **lrDiffRef**: Float - Left-right balance difference in mm
- **fbDiffRef**: Float - Front-back balance difference in mm

### Examples
```
PID,0,250,0,1000\n         # Set Repelling: Kp=250, Ki=0, Kd=1000
REF,11.0,-2.0,0.0\n        # Set avgRef=11mm, LRDiff=-2mm, FBDiff=0mm
PID,3,0,0,100\n            # Set RollLeftUp: Kp=0, Ki=0, Kd=100
REF,12.5,0.0,0.5\n         # Set avgRef=12.5mm, LRDiff=0mm, FBDiff=0.5mm
```

## Implementation Details

### Arduino Side (AdditiveControlCode.ino)

#### Default Values
```cpp
Constants repelling = {250, 0, 1000};
Constants attracting = {250, 0, 1000};
Constants RollLeftUp = {0, 0, 100};
Constants RollLeftDown = {0, 0, 100};
Constants RollFrontUp = {0, 0, 500};
Constants RollFrontDown = {0, 0, 500};
```

#### Serial Command Processing
The main loop now parses incoming serial commands:
- Commands starting with "PID," are parsed for PID tuning
- Single character commands ('0' or '1') control the system on/off state
- Upon receiving a PID command, the corresponding Constants struct is updated
- The controller's internal PID values are updated via setter methods

### Controller (Controller.hpp/cpp)

#### New Methods
```cpp
// PID update methods
void updateAvgPID(Constants repel, Constants attract);
void updateLRPID(Constants down, Constants up);
void updateFBPID(Constants down, Constants up);

// Reference update method
void updateReferences(float avgReference, float lrDiffReference, float fbDiffReference);
```

These methods update the controller's internal values:
- **PID Constants:**
  - `avgConsts` - Controls average height (repelling/attracting)
  - `LConsts` - Controls left-right balance (RollLeftDown/RollLeftUp)
  - `FConsts` - Controls front-back balance (RollFrontDown/RollFrontUp)
- **Reference Values:**
  - `AvgRef` - Target average height
  - `LRDiffRef` - Target left-right balance difference
  - `FBDiffRef` - Target front-back balance difference

### Python GUI (serial_plotter.py)

#### Features
- **Individual PID Entry**: Each control mode has dedicated Kp, Ki, Kd input fields
- **Reference Value Controls**: Set avgRef, LRDiffRef, FBDiffRef targets
- **Send Button**: Each mode has its own "Send" button for individual updates
- **Send All**: A "Send All PID Values" button transmits all 6 modes at once
- **Send References**: Update all reference values with one click
- **Default Values**: GUI is pre-populated with the default values from the Arduino code
- **Scrollable Interface**: All PID controls are in a scrollable panel for easy access

#### Control Flow - PID Updates
1. User enters PID values in the GUI
2. Clicks "Send" for individual mode or "Send All" for batch update
3. Python formats the command: `PID,<mode>,<kp>,<ki>,<kd>\n`
4. Command is sent via serial to Arduino
5. Arduino parses the command and updates the controller
6. Changes take effect immediately in the control loop

#### Control Flow - Reference Updates
1. User enters reference values in the GUI (Avg Height, LR Diff, FB Diff)
2. Clicks "Send References"
3. Python formats the command: `REF,<avgRef>,<lrDiffRef>,<fbDiffRef>\n`
4. Command is sent via serial to Arduino
5. Arduino updates the reference variables and controller
6. New setpoints take effect immediately

## Usage Instructions

### Starting the GUI
```bash
python serial_plotter.py --port /dev/cu.usbmodemXXXX
```

Or for testing without hardware:
```bash
python serial_plotter.py --mock
```

### Tuning Workflow
1. **Start the application** - GUI launches with current PID values
2. **Connect to Arduino** - Serial connection established automatically
3. **Adjust values** - Modify Kp, Ki, Kd for desired control mode
4. **Send to Arduino** - Click "Send" for that mode or "Send All"
5. **Observe behavior** - Watch real-time plots to see the effect
6. **Iterate** - Adjust and resend as needed

### Tips for Tuning
- Start with one mode at a time
- Use "Send All" to load a complete tuning set
- Watch the real-time plots to observe the effect of changes
- The system updates immediately - no restart required

## Data Flow Summary

**PID Updates:**
```
GUI Input → Serial Command → Arduino Parse → Constants Update → 
Controller Setter → Internal PID Update → Control Loop → PWM Output
```

**Reference Updates:**
```
GUI Input → Serial Command → Arduino Parse → Reference Variables Update → 
Controller Setter → Internal Reference Update → Control Loop → PWM Output
```

## Compatibility Notes

- The GUI requires `tkinter` (included with most Python installations)
- Serial communication uses standard pyserial library
- Arduino code uses String parsing (ensure sufficient memory)
- Commands are processed in the main loop between control cycles

## Troubleshooting

### GUI doesn't send commands
- Check serial connection
- Verify port selection
- Look for error messages in console

### Arduino doesn't respond
- Ensure Arduino code is uploaded
- Check Serial Monitor for confirmation messages
- Verify baud rate (115200)

### Values don't take effect
- Confirm command format is correct
- Check that mode number (0-5) is valid
- Ensure values are within reasonable ranges

## Future Enhancements

Possible additions:
- Save/load PID profiles to/from file
- Preset tuning configurations
- Auto-tuning algorithms
- Real-time response visualization
- Logging of PID changes with timestamps
