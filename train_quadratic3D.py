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
from mpl_toolkits.mplot3d import Axes3D


print("Compiling loma code...")
try:
    with open("KAN_Implmentation/kan.py", "r") as f:
        # Compile to C, outputting a shared library (_code/kan.so on Linux/macOS, _code/kan.dll on Windows)
        structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan")
    print("Compilation successful!")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()


# random_numbers = np.random.rand(100)
# X_train = np.random.uniform(low=0, high=1, size=100)


# X_test = np.random.uniform(low=0, high=1, size=50)

n = 100 # Number of rows
m = 2 # Number of columns

# --- Option 1: Random floats between 0.0 (inclusive) and 1.0 (exclusive) ---
X_train = np.random.rand(n, m,)

X_test = np.random.rand(50, m)

# Calculate the y_true: 2x + 1
# y_true = X_train[:, 0]**2 + X_train[:, 1]**2

def f(x):
   return np.sin(x[:, 0]) + np.cos(x[:, 1]) 
    # return .5*x[:, 0]**2 + x[:, 1]**2


y_true = f(X_train)

n_inputs, n_outputs = 2, 1
total_control_points = n_inputs * n_outputs * 3
control_points = np.random.rand(total_control_points).astype(np.float32) * 2 - 1 # Random init between -1 and 1


EPOCHS = 25

X_plot = np.arange(EPOCHS)
y_loss = np.zeros(EPOCHS)

num_control_points = len(control_points)

def calc_loss(y_true, y_pred):
    return (y_true - y_pred)**2

def d_calc_loss(y_true, y_pred):
    return 2 * (y_true - y_pred) * -1



learning_rate = .003

for epoch in range(EPOCHS):
    grads = np.zeros_like(control_points)
    acc_loss = 0.0

    InputsArrayType = ctypes.c_float * n_inputs
    OutputsArrayType = ctypes.c_float * n_outputs
    CPsArrayType = ctypes.c_float * total_control_points
    
    buff_type = ctypes.c_float * 3
    for i in range(len(X_train)):
        x_val = X_train[i]
        y_val = y_true[i]

        c_inputs = InputsArrayType(*x_val)
        c_control_points_in = CPsArrayType(*control_points)
        c_outputs = OutputsArrayType(*np.zeros(n_outputs))
        buff = buff_type(*np.zeros(3))
        d_buff = buff_type(*np.zeros(3))
        try:
            # --- Forward Pass for KAN Layer ---
            
            lib.kan_layer(c_inputs, c_control_points_in, ctypes.c_int(n_inputs), ctypes.c_int(n_outputs), buff, c_outputs)
            print(list(c_outputs))

            y_pred = np.array(list(c_outputs))

            loss = np.sum(calc_loss(y_val, y_pred))
            acc_loss += loss

            # --- Backward Pass for KAN Layer ---
            d_outputs = OutputsArrayType(*d_calc_loss(y_val, y_pred).flatten())
            
            c_d_inputs_current_sample = InputsArrayType(*np.zeros(n_inputs))
            c_d_control_points_current_sample = CPsArrayType(*np.zeros_like(control_points))

            lib.d_kan_layer(c_inputs, c_d_inputs_current_sample, c_control_points_in, c_d_control_points_current_sample, ctypes.c_int(n_inputs), ctypes.c_int(n_inputs), ctypes.c_int(n_outputs),  ctypes.c_int(n_outputs), buff, d_buff, d_outputs)
            
            grads += np.array(list(c_d_control_points_current_sample))

        except Exception as e:
            print(f"Error during KAN layer pass for sample {i}: {e}")
            sys.exit(1)

    control_points -= learning_rate * grads
    y_loss[epoch] = acc_loss

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"EPOCH: {epoch}, Loss: {acc_loss:.4f}")
    
print(f"final control_points: {control_points}")

y_test_true = f(X_test)

test = []

print("TESTING")
for x_test, y_test in zip(X_test, y_test_true):
    print(f"passing in ({x_test[0]}, {x_test[1]}) {y_test})")

    c_inputs = InputsArrayType(*x_test)
    c_control_points_in = CPsArrayType(*control_points)
    c_outputs = OutputsArrayType(*np.zeros(n_outputs))
    buff = buff_type(*np.zeros(3))
    d_buff = buff_type(*np.zeros(3))

    lib.kan_layer(c_inputs, c_control_points_in, ctypes.c_int(n_inputs), ctypes.c_int(n_outputs), buff, c_outputs)

    y_pred = list(c_outputs)[0]
    test.append(y_pred)
        
    print(f"Predicted: {y_pred}, True: {y_test}")
     

plt.figure(figsize=(10, 6))
plt.plot(X_plot, y_loss, label='Total Loss per Epoch', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.grid(True)
plt.legend()
plt.show()



# fig = plt.figure(figsize=(16, 8)) # Increased width for side-by-side plots

# ax1 = fig.add_subplot(1, 2, 1, projection='3d') # 1 row, 2 columns, 1st subplot
# ax1.scatter(X_train[:, 0], X_train[:, 1], y_true, c='green', s=10)
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('Training Data')
# ax1.view_init(elev=20, azim=-45) # Keep original view or adjust as needed

# ax2 = fig.add_subplot(1, 2, 2, projection='3d') # 1 row, 2 columns, 2nd subplot
# ax2.scatter(X_test[:, 0], X_test[:, 1], y_test_true, c='red', s=10, label='True')
# ax2.scatter(X_test[:, 0], X_test[:, 1], test, c='blue', s=10, label='Predicted')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title('Test Data')
# ax2.legend()
# ax2.view_init(elev=20, azim=-45) # Keep original view or adjust as needed

# fig.suptitle("Comparison of Training and Test Data z = sin(x) + cos(y)", fontsize=16)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect if suptitle is used
# plt.show()