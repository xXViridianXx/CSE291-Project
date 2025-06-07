import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
import compiler
import ctypes
import error
import math
import gpuctypes.opencl as cl
import cl_utils
import unittest
import numpy as np

try:
    with open("KAN_Implmentation/kan.py", "r") as f:
        structs, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan")
    print("Compilation successful!")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()
    
# Test Case 1: x is exactly at a knot
x_test_1 = .2
control_points_test_1 = np.array([-.1, .5], dtype=np.float32)
# Expected: For x=2.0, u=0.0, output = cp[0]*(1-0) + cp[1]*0 = 10.0

# Test Case 2: x is halfway between knots
x_test_2 = 2.5
control_points_test_2 = np.array([10.0, 20.0], dtype=np.float32)
# Expected: For x=2.5, u=0.5, output = cp[0]*(0.5) + cp[1]*(0.5) = 10.0*0.5 + 20.0*0.5 = 5.0 + 10.0 = 15.0

# Test Case 3: x is near the end of a segment
x_test_3 = 2.9
control_points_test_3 = np.array([10.0, 20.0], dtype=np.float32)
# Expected: For x=2.9, u=0.9, output = cp[0]*(0.1) + cp[1]*(0.9) = 10.0*0.1 + 20.0*0.9 = 1.0 + 18.0 = 19.0

# Test Case 4: Negative x value
x_test_4 = -0.5
control_points_test_4 = np.array([-5.0, 5.0], dtype=np.float32)
# Expected: For x=-0.5, knot_index=-1, u=-0.5 - (-1.0) = 0.5. Output = -5.0*(0.5) + 5.0*(0.5) = -2.5 + 2.5 = 0.0


test_cases = [
    (x_test_1, control_points_test_1, "Test Case 1 (x=2.0)"),
    (x_test_2, control_points_test_2, "Test Case 2 (x=2.5)"),
    (x_test_3, control_points_test_3, "Test Case 3 (x=2.9)"),
    (x_test_4, control_points_test_4, "Test Case 4 (x=-0.5)"),
]

# --- 3. Run Test Cases ---
for i, (x_val, cp_val, description) in enumerate(test_cases):
    print(f"\n--- {description} ---")
    print(f"Input x: {x_val}")
    print(f"Control Points: {cp_val}")

    c_x = ctypes.c_float(x_val)
    CPArrayType = ctypes.c_float * len(cp_val)
    c_control_points = CPArrayType(*cp_val)
    c_output = ctypes.c_float(0.0)
    
    d_c_x = ctypes.c_float(x_val)
    test = np.zeros(2)
    d_c_control_points = CPArrayType(*test)
    try:
        lib.linear_bspline(c_x, c_control_points, c_output)
        result = c_output.value
        print(f"Loma Output: {result}")
        
        
        d_loss = ctypes.c_float(-2.76) 
        lib.d_linear_bspline(c_x, d_c_x, c_control_points, d_c_control_points, d_loss)
         
        print(f" c_x: {c_x} | d_c_x : {d_c_x}")
        print(f"c_control_points: {list(c_control_points)}| d_c_control_points : {list(d_c_control_points)}")
        knot_index_manual = int(np.floor(x_val))
        u_manual = x_val - knot_index_manual
        expected_output = cp_val[0] * (1.0 - u_manual) + cp_val[1] * u_manual
        print(f"Expected Output (Manual): {expected_output}")

        if np.isclose(result, expected_output):
            print("Result matches expected. PASSED!")
        else:
            print("Result DOES NOT match expected. FAILED!")

    except Exception as e:
        print(f"Function call failed for test case: {e}")



