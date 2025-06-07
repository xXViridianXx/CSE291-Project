import numpy as np
import unittest
import cl_utils
import gpuctypes.opencl as cl
import math
import error
import ctypes
import compiler
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


import matplotlib.pyplot as plt


print("Compiling loma code...")
try:
    with open("KAN_Implmentation/kan.py", "r") as f:
        # Compile to C, outputting a shared library (_code/kan.so on Linux/macOS, _code/kan.dll on Windows)
        structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan")
    print("Compilation successful!")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()


X_train = np.array([.2, .4, .6, .8])

# Calculate the y_true: 2x + 1
y_true = 2 * X_train + 1

control_points = np.array([-.1, .5], dtype=np.float32)

EPOCHS = 150

X_plot = np.arange(EPOCHS)
y_loss = np.zeros(EPOCHS)

num_control_points = len(control_points)

def calc_loss(y_true, y_pred):
    return (y_true - y_pred)**2

def d_calc_loss(y_true, y_pred):
    return 2 * (y_true - y_pred) * -1



learning_rate = .01

for epoch in range(EPOCHS):
    grads = np.zeros_like(control_points)
    acc_loss = 0
    for x_val, y_val in zip(X_train, y_true):
        print(f"passing in ({x_val}. {y_val})")
        c_x = ctypes.c_float(x_val)
        CPArrayType = ctypes.c_float * num_control_points
        c_control_points = CPArrayType(*control_points)

        c_output = ctypes.c_float(0.0)
        d_c_x = ctypes.c_float(x_val)
        # Define ArrayType dynamically based on cp_val length
        # d_c_control_points = ctypes.c_float * len(cp_val)
        test = np.zeros(2)
        d_c_control_points = CPArrayType(*test)
        # Call the compiled linear_bspline function
        
        try:
            lib.linear_bspline(c_x, c_control_points, c_output)
            y_pred = c_output.value
            print(f"Predicted: {y_pred}")
            
            loss = calc_loss(y_val, y_pred)
            
            acc_loss += loss
            
            d_loss = ctypes.c_float(d_calc_loss(y_val, y_pred))
            
            print(f"loss: {loss} | d_loss: {d_loss}") 
            lib.d_linear_bspline(c_x, d_c_x, c_control_points, d_c_control_points, d_loss)
            print(f"c_control_points: {list(c_control_points)}| d_c_control_points : {list(d_c_control_points)}")
            
            grads += np.array(d_c_control_points)
            

        except Exception as e:
            print(f"Function call failed for this test case: {e}")
            
    y_loss[epoch] = acc_loss
    control_points -= learning_rate * grads
    print(f"control_points: {control_points}")
    print(f"EPOCH: {epoch}\n")
    
print(f"final control_points: {control_points}")


X_test = np.array([0, .3, .37, .7, .75])
y_test_true = 2 * X_test + 1
 
print("TESTING")
for x_test, y_test in zip(X_test, y_test_true):
    print(f"passing in ({x_test}, {y_test})")

    c_x = ctypes.c_float(x_test)
    CPArrayType = ctypes.c_float * num_control_points
    c_control_points = CPArrayType(*control_points)

    c_output = ctypes.c_float(0.0)

    lib.linear_bspline(c_x, c_control_points, c_output)
    y_pred = c_output.value
        
        
    print(f"Predicted: {y_pred}, True: {y_test}")

plt.figure(figsize=(10, 6)) # Create a figure and set its size
plt.plot(X_plot, y_loss, label='Total Loss per Epoch', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.grid(True) # Add a grid for better readability
plt.legend() # Show the legend
plt.show() # Display the plot