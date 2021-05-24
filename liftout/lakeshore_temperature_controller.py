import serial
import io
import time

def get_temperature(ser):
    command = "KRDG? A\r\n"
    ser.write(bytes(command,'utf-8'))
    time.sleep(0.1)
    ser.flush()
    output = ser.read_all()
    output = float( output[:-2] )
    output = output - 273
    print(output)
    return output



ser = serial.Serial(port='COM17', baudrate=57600, bytesize=7, stopbits=1, parity=serial.PARITY_ODD, timeout=1)
ser.open()

#command = "*IDN?\r\n"
#command = "TEMP?\r\n" # Temperature is in K. returns the T of the ceramic thermocouple block used in the room temperature compensation calculation.
#command = "RANGE? 1\r\n"
#command = "WARMUP? 1\r\n"
#command = "SETP 1,40\r\n"
#command = "PID? 1\r\n"
command = "PID 1,500,20,0\r\n"
ser.write(bytes(command,'utf-8'))
time.sleep(0.1)
ser.flush()
output = ser.read_all()
print(output)







command = "TLIMIT A,373\r\n"
ser.write(bytes(command,'utf-8'))
time.sleep(0.1)
ser.flush()
output = ser.read_all()
print(output)

command = "WARMUP 0,50\r\n"
ser.write(bytes(command,'utf-8'))
time.sleep(0.1)
ser.flush()
output = ser.read_all()
print(output)

set_temperature = -140
command = "SETP 1," + str(set_temperature) + "\r\n"
ser.write(bytes(command,'utf-8'))
time.sleep(0.1)
ser.flush()
output = ser.read_all()
print(output)

command = "RANGE 1,1\r\n"
ser.write(bytes(command,'utf-8'))
time.sleep(0.1)
ser.flush()
output = ser.read_all()
print(output)


while 1:
    temperature = get_temperature(ser)
    if temperature >= set_temperature - 0.5:
        command = "RANGE 1,0\r\n"  # turn of the heater
        ser.write(bytes(command,'utf-8'))
        time.sleep(0.1)
        ser.flush()
        output = ser.read_all()
        print(output)
        break
    time.sleep(5)


'''
*OPC? Operation Complete Query

Warm Up Supply mode is only available when Output 2 is in Voltage mode. The Control Input setting, determines the sensor that is used for feedback in the Warm Up
Supply mode. Refer to section 4.5.1.7.1 for details on the Control Input parameter and
section 4.5.1 for Output Type.
1) Once Warm Up Supply mode is configured,
2) use the Setpoint key to set the desiredtemperature,
3) then use the Heater Range key to activate the output by setting the range to On.
The front panel display must be configured to show the Warm Up control
loop for the Setpoint and Heater Range keys to be used. Refer to section 4.2 and
section 4.3 for details on front panel keypad operation and display setup.

RANGE Heater Range Command
Input RANGE <output>,<range>[term]
Format n,n
<output> Specifies which output to configure: 1 or 2.
<range> For Outputs 1 and 2 in Current mode: 0 = Off, 1 = Low, 2 = Medium, 3 = High
For Output 2 in Voltage mode: 0 = Off, 1 = On
Remarks The range setting has no effect if an output is in the Off mode, and does not apply to
an output in Monitor Out mode. An output in Monitor Out mode is always on.

WARMUP Warmup Supply Parameter Command
WARMUP <output>,<control>,<percentage>[term]
<output> Output 2 is the only valid entry and must be included.
<control> Specifies the type of control used: 0 = Auto Off, 1 = Continuous
<percentage> Specifies the percentage of full scale (10 V) Monitor Out
Example: WARMUP 1,50[term] Output 2 in Voltage mode will use continuous control, with a 5 V (50.50%) output voltage for activating the external power supply.

SETP Control Setpoint Command
Input SETP <output>,<value>[term]
Format n,±nnnnnn
<output> Specifies which output’s control loop to configure: 1 or 2.
<value> The value for the setpoint (in the preferred units of the
control loop sensor).
Example SETP 1,122.5[term] Output 1 setpoint is now 122.5 (based on its units).
Remarks Control settings, that is, P, I, D, and Setpoint, are assigned to outputs, which results in
the settings being applied to the control loop formed by the output and its control
input

SETP? Control Setpoint Query
Input SETP? <output>[term]



'''