# forward_euler: 1D spatial comparison

| dim | method | Nx | dx | dt | Nt | T | runtime_sec | stability_safe | min_u | max_u | mean_u | AZ_Relative_L2_Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D | forward_euler | 101 | 0.400 | 0.005 | 2000 | 10.0 | 0.158753 | True | 0.293756 | 1.000000 | 0.931159 | 0.001012 |
| 1D | forward_euler | 201 | 0.200 | 0.005 | 2000 | 10.0 | 0.161578 | True | 0.293756 | 1.000000 | 0.932478 | 0.001318 |
| 1D | forward_euler | 401 | 0.100 | 0.005 | 2000 | 10.0 | 0.166832 | True | 0.293756 | 1.000000 | 0.933166 | 0.001396 |
