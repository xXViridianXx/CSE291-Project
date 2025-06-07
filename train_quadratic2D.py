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


try:
    with open("KAN_Implmentation/kan.py", "r") as f:
        structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan")
    print("Compilation successful!")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()


X_train = np.random.uniform(low=0, high=1, size=100)


X_test = np.random.uniform(low=0, high=1, size=50)

def f(x):
    return 3 * x**2 - 2 * x + 1

y_true = f(X_train)

control_points = np.array([-.1, .5, .8], dtype=np.float32)

EPOCHS = 25

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
        test = np.zeros(3)
        d_c_control_points = CPArrayType(*test)
        
        try:
            lib.quadratic_bspline(c_x, c_control_points, c_output)
            y_pred = c_output.value
            
            
            print(f"Predicted: {y_pred}")
            
            loss = calc_loss(y_val, y_pred)
            
            acc_loss += loss
            
            d_loss = ctypes.c_float(d_calc_loss(y_val, y_pred))
            
            print(f"loss: {loss} | d_loss: {d_loss}") 
            lib.d_quadratic_bspline(c_x, d_c_x, c_control_points, d_c_control_points, d_loss)
            print(f"c_control_points: {list(c_control_points)}| d_c_control_points : {list(d_c_control_points)}")
            
            grads += np.array(d_c_control_points)
            

        except Exception as e:
            print(f"Function call failed for this test case: {e}")
            
    y_loss[epoch] = acc_loss
    control_points -= learning_rate * grads
    print(f"control_points: {control_points}")
    print(f"EPOCH: {epoch}\n")
    
print(f"final control_points: {control_points}")

y_test_true = f(X_test)
 
print("TESTING")

test = []
for x_test, y_test in zip(X_test, y_test_true):
    print(f"passing in ({x_test}. {y_test})")

    c_x = ctypes.c_float(x_test)
    CPArrayType = ctypes.c_float * num_control_points
    c_control_points = CPArrayType(*control_points)

    c_output = ctypes.c_float(0.0)

    lib.quadratic_bspline(c_x, c_control_points, c_output)
    y_pred = c_output.value
    
    test.append(y_pred)
        
    print(f"Predicted: {y_pred}, True: {y_test}")
     

# plt.figure(figsize=(10, 6)) # Create a figure and set its size
# plt.plot(X_plot, y_loss, label='Total Loss per Epoch', color='blue')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Total Loss')
# plt.grid(True) # Add a grid for better readability
# plt.legend() # Show the legend
# plt.show() # Display the plot

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 9))
ax1.scatter(X_train, y_true, marker='o', s=10, alpha=0.7, c="red", label="Training Points")
ax1.set_title("Training Data")
ax1.set_ylabel("y")
ax1.grid(True)

ax2.scatter(X_test, y_test_true, marker='o', s=10, alpha=0.7, c="blue", label="True Points")
ax2.set_title("Test Data")
ax2.set_ylabel("y")
ax2.grid(True)

ax2.scatter(X_test, test, marker='o', s=10, alpha=0.7, c="green", label="Predicted Points")
ax2.set_title("Predicted ")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid(True)
ax2.legend()

fig.suptitle("Comparison of Training and Test Data for $f(x) = 3x^2 - 2x + 1$")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
