# forward_euler: 1D temporal reference convergence

| dim | method | Nx | dx | dt | dt_ref | Nt | T | runtime_sec | Reference_Relative_L2_Error | Observed_Time_Order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D | forward_euler | 201 | 0.200 | 0.02 | 0.0013 | 500 | 10.0 | 0.039483 | 0.005408 | 1.114 |
| 1D | forward_euler | 201 | 0.200 | 0.01 | 0.0013 | 1000 | 10.0 | 0.082297 | 0.002499 | 1.230 |
| 1D | forward_euler | 201 | 0.200 | 0.005 | 0.0013 | 2000 | 10.0 | 0.160238 | 0.001066 | 1.589 |
| 1D | forward_euler | 201 | 0.200 | 0.0025 | 0.0013 | 4000 | 10.0 | 0.297786 | 0.000354 |  |
