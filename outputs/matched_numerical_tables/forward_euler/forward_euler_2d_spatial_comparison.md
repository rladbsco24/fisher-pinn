# forward_euler: 2D spatial comparison

| dim | method | Nx | Ny | dx | dy | dt | Nt | T | runtime_sec | stability_safe | min_U | max_U | mean_U | Exact_Relative_L2_Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D | forward_euler | 41 | 41 | 0.750 | 0.750 | 0.01 | 300 | 3.0 | 0.077171 | True | 0.000004 | 0.999972 | 0.643876 | 0.001245 |
| 2D | forward_euler | 61 | 61 | 0.500 | 0.500 | 0.01 | 300 | 3.0 | 0.127027 | True | 0.000004 | 0.999972 | 0.644827 | 0.001227 |
| 2D | forward_euler | 81 | 81 | 0.375 | 0.375 | 0.01 | 300 | 3.0 | 0.184513 | True | 0.000004 | 0.999972 | 0.645310 | 0.001223 |
