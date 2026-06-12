# forward_euler: 2D temporal reference convergence

| dim | method | Nx | Ny | dx | dy | dt | dt_ref | Nt | T | runtime_sec | Reference_Relative_L2_Error | Observed_Time_Order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D | forward_euler | 81 | 81 | 0.375 | 0.375 | 0.01 | 0.0013 | 300 | 3.0 | 0.201821 | 0.001060 | 1.220 |
| 2D | forward_euler | 81 | 81 | 0.375 | 0.375 | 0.005 | 0.0013 | 600 | 3.0 | 0.381264 | 0.000455 | 1.584 |
| 2D | forward_euler | 81 | 81 | 0.375 | 0.375 | 0.0025 | 0.0013 | 1200 | 3.0 | 0.694913 | 0.000152 |  |
