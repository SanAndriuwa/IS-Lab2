# IS-Lab2 MATLAB Solution

Assignment: [serackis/IS-Lab2](https://github.com/serackis/IS-Lab2).

- `lab2_main.m`: main task, 20 examples, a 1–8–1 network, tanh hidden layer, linear output, and manual backpropagation.
- `lab2_surface.m`: additional task, two-variable surface, and a 2–8–1 network.
- `DEFENSE.md`: explanations, formulas, and defense questions.

Open an `.m` file in MATLAB and press Run. The data is created inside the script. No extra toolboxes are required. The script prints coefficients and errors and creates plots.

The MATLAB R2026a static analyzer reported no issues.

The unmatched parentheses in the README formula were corrected: the complete sum `1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)` is divided by 2. The additional task uses the surface `0.5 + 0.25*sin(pi*x1)*cos(pi*x2)`.

The main network has a limit of 100,000 epochs and the surface network has a limit of 10,000. Both stop earlier when MSE < 0.0001. If a limit is reached, use the printed MSE as the result.

An independent numerical check gave MSE values of about 0.000100 and 0.000107. These are not MATLAB results: MATLAB stopped before execution in this environment with `File system inconsistency`. Python uses different initialization, so MATLAB weights and epoch counts may differ. Run the scripts in MATLAB before the defense.

