def linear_bspline(x: In[float], control_points: In[Array[float]], output: Out[float]):
    
    knot_index: int = float2int(x) 
    if x < 0:
        knot_index = float2int(x - 1.0)
    
    t: float = x - int2float(knot_index)
    output = (1 - t) * control_points[0] + t * control_points[1]


def quadratic_bspline(x: In[float], control_points: In[Array[float]], output: Out[float]):

    knot_index: int = float2int(x) 
    if x < 0:
        knot_index = float2int(x - 1.0)
        
    t: float = x - int2float(knot_index)
     
    p1: float = control_points[knot_index]
    p2: float = control_points[knot_index + 1]
    p3: float = control_points[knot_index + 2]

    q1: float = (1 - t) * p1 + t * p2
    q2: float = (1 - t) * p2 + t * p3
    
    output = (1 - t) * q1 + t * q2    

def kan_layer(inputs: In[Array[float]], control_points: In[Array[float]], n_inputs: In[int], 
              n_outputs: In[int], buffer: In[Array[float]], outputs: Out[Array[float]]):
    
    out_neuron: int
    in_neuron: int
    summed_phis: float
    k: int
    temp: float
    control_point_offset: int
    inp_val: float

    while (out_neuron < n_outputs, max_iter := 10):
        summed_phis = 0
        while (in_neuron < n_inputs, max_iter:= 10):
            
            control_point_offset = (in_neuron * n_outputs + out_neuron) * 3
            
            k = 0
            while (k < 3, max_iter := 3):
                buffer[k] = control_points[control_point_offset + k]
                k = k + 1
            
            inp_val = inputs[in_neuron]
            quadratic_bspline(inp_val, buffer, temp)
            
            
            summed_phis = summed_phis + temp
            in_neuron = in_neuron + 1
            outputs[out_neuron] = summed_phis
        
        out_neuron = out_neuron + 1
        
        outputs[out_neuron] = summed_phis
        
d_linear_bspline = rev_diff(linear_bspline)
d_quadratic_bspline = rev_diff(quadratic_bspline)
d_kan_layer = rev_diff(kan_layer)