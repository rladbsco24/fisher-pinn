# forward_euler: 1D time comparison

| dim | method | Nx | dx | dt | Nt | T | runtime_sec | stability_safe | min_u | max_u | mean_u | AZ_Relative_L2_Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1D | forward_euler | 201 | 0.200 | 0.02 | 500 | 10.0 | 0.045558 | False | 0.293756 | 1.000000 | 0.930742 | 0.005660 |
| 1D | forward_euler | 201 | 0.200 | 0.01 | 1000 | 10.0 | 0.084172 | True | 0.293756 | 1.000000 | 0.931905 | 0.002752 |
| 1D | forward_euler | 201 | 0.200 | 0.005 | 2000 | 10.0 | 0.147941 | True | 0.293756 | 1.000000 | 0.932478 | 0.001318 |
| 1D | forward_euler | 201 | 0.200 | 0.0025 | 4000 | 10.0 | 0.373045 | True | 0.293756 | 1.000000 | 0.932763 | 0.000607 |
