.. code:: ipython3

    # -----------------------------------------------
    # Deutsch-Jozsa Algorithm – Clean vs Malicious Dataset (Qiskit Simulation)
    # -----------------------------------------------
    
    # This notebook implements the Deutsch-Jozsa (DJ) algorithm using Qiskit,
    # simulating both clean and malicious quantum circuits to create a dataset
    # for anomaly or Trojan detection in quantum environments.
    
    # Objective:
    # - Generate 10 clean Deutsch-Jozsa circuits with valid constant/balanced oracles
    # - Generate 10 malicious DJ circuits with subtle tampering or Trojan-like behaviors
    # - Simulate all circuits using Qiskit backends and extract rich features
    
    # Dataset Composition:
    # - Clean class labeled as 0
    # - Malicious class labeled as 1
    # - Each sample includes structural (gate counts, depth) and measurement (success rate, entropy) features
    
    # Visualization & Analysis:
    # - Circuit diagrams and output histograms are saved for each variant
    # - Class balance visualization
    # - Correlation heatmap of extracted features
    
    # Tools & Environment:
    # - Qiskit for circuit generation, transpilation, and simulation
    # - matplotlib and seaborn for visualization
    # - pandas and numpy for data handling
    
    # This end-to-end notebook supports Quantum ML research in secure quantum computation.
    
    # Author: Zeeshan Ajmal
    

.. code:: ipython3

    # Core Qiskit imports
    from qiskit import QuantumCircuit, transpile, quantum_info
    from qiskit import schedule as build_schedule
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as norm_sampler
    
    
    # Visualization tools
    %matplotlib inline
    from qiskit.visualization import plot_histogram, plot_circuit_layout
    import matplotlib.pyplot as plt
    
    # Python utilities
    import seaborn as sns
    import functools
    import os
    import numpy as np
    

.. code:: ipython3

    # -----------------------------------------------
    # Initializing IBM Quantum Runtime Account
    # -----------------------------------------------
    
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token='ff1dde434d0dcec5608d0b0166f3df690e5b8258114b55d50805fe2c5c25d03a520f7551cd363d5295b421ab19908870fc00cdfb57a074f0a7eaa6c9ff2fa9e6'
    )

.. code:: ipython3

    def generate_dj_clean_variation(index):
        """
        Generate a clean Deutsch-Jozsa quantum circuit.
        Each variation alternates between constant and balanced oracles.
        """
        n = 3  # number of input qubits
        qc = QuantumCircuit(n + 1)  # +1 for the output qubit
    
        # Step 1: Initialize output qubit to |1⟩ and apply Hadamard
        qc.x(n)
        qc.h(range(n + 1))
    
        # Step 2: Apply Oracle
        if index % 2 == 0:
            # Constant oracle (always 0)
            pass  # identity oracle, do nothing
        elif index % 4 == 1:
            # Constant oracle (always 1)
            qc.x(n)
        elif index % 4 == 3:
            # Balanced oracle (XOR of two qubits)
            qc.cx(0, n)
            qc.cx(1, n)
        else:
            # Balanced oracle (single XOR)
            qc.cx(2, n)
    
        # Step 3: Apply Hadamard to input qubits
        qc.h(range(n))
    
        # Step 4: Measure input qubits only
        qc.measure_all()
    
        return qc
    

.. code:: ipython3

    # Dictionary to hold clean DJ circuits
    dj_clean_circuits = {}
    
    # Generate 10 clean variations
    for i in range(10):
        circuit = generate_dj_clean_variation(i)
        dj_clean_circuits[f"dj_clean_{i}"] = circuit

.. code:: ipython3

    # --------------------------------------------------
    # Simulate, Visualize, and Store Clean DJ Circuit Outputs
    # --------------------------------------------------
    
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as norm_sampler
    
    # Initialize Qiskit Runtime Service (if not already done)
    service = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q/open/main')
    
    # Set preferred backend
    backend = service.backend("ibm_brisbane")
    
    # Create folders for visuals
    os.makedirs("dj_clean_diagrams", exist_ok=True)
    os.makedirs("dj_clean_histograms", exist_ok=True)
    
    results_summary_clean = []
    
    for i in range(10):
        circuit_name = f"dj_clean_{i}"
        circuit = dj_clean_circuits[circuit_name]
    
        # Transpile & run
        transpiled = transpile(circuit, backend=backend)
        sampler = norm_sampler(backend)
        job = sampler.run([transpiled])
        result = job.result()
    
        # Extract counts and success info
        counts = result[0].data.meas.get_counts()
        success_state = max(counts, key=counts.get)
        percentage = counts[success_state] / sum(counts.values()) * 100
    
        # Save circuit diagram
        circuit.draw("mpl", filename=f"dj_clean_diagrams/{circuit_name}.png")
    
        # Save histogram
        fig = plot_histogram(counts, title=f"{circuit_name} | {percentage:.2f}% success")
        fig.savefig(f"dj_clean_histograms/{circuit_name}_histogram.png")
        plt.show()
    
        # Store result summary
        results_summary_clean.append({
            "name": circuit_name,
            "success_state": success_state,
            "success_rate": percentage
        })
    
    # Display results
    for res in results_summary_clean:
        print(f"{res['name']} | State: {res['success_state']} | Success Rate: {res['success_rate']:.2f}%")



.. image:: output_5_0.png



.. image:: output_5_1.png



.. image:: output_5_2.png



.. image:: output_5_3.png



.. image:: output_5_4.png



.. image:: output_5_5.png



.. image:: output_5_6.png



.. image:: output_5_7.png



.. image:: output_5_8.png



.. image:: output_5_9.png



.. image:: output_5_10.png



.. image:: output_5_11.png



.. image:: output_5_12.png



.. image:: output_5_13.png



.. image:: output_5_14.png



.. image:: output_5_15.png



.. image:: output_5_16.png



.. image:: output_5_17.png



.. image:: output_5_18.png



.. image:: output_5_19.png


.. parsed-literal::

    dj_clean_0 | State: 0000 | Success Rate: 50.15%
    dj_clean_1 | State: 1000 | Success Rate: 49.95%
    dj_clean_2 | State: 1000 | Success Rate: 51.34%
    dj_clean_3 | State: 0011 | Success Rate: 47.31%
    dj_clean_4 | State: 1000 | Success Rate: 49.90%
    dj_clean_5 | State: 1000 | Success Rate: 51.64%
    dj_clean_6 | State: 1000 | Success Rate: 50.34%
    dj_clean_7 | State: 1011 | Success Rate: 47.46%
    dj_clean_8 | State: 1000 | Success Rate: 50.46%
    dj_clean_9 | State: 1000 | Success Rate: 51.25%
    

.. code:: ipython3

    # --------------------------------------------------
    # Feature Extraction for Deutsch-Jozsa Circuits
    # --------------------------------------------------
    
    def extract_features(circuit_name, circuit, result, label):
        # Structural features
        ops = circuit.count_ops()
        num_cx = ops.get('cx', 0)
        num_h = ops.get('h', 0)
        num_x = ops.get('x', 0)
        num_ccx = ops.get('ccx', 0)
        depth = circuit.depth()
        total_gates = sum(ops.values())
    
        # Measurement-based features
        counts = result[0].data.meas.get_counts()
        probs = np.array(list(counts.values()))
        probs = probs / probs.sum()
    
        success_state = max(counts, key=counts.get)
        success_rate = counts[success_state] / sum(counts.values()) * 100
        entropy = shannon_entropy(probs, base=2)
        unique_states = len(counts)
    
        return {
            "name": circuit_name,
            "depth": depth,
            "cx": num_cx,
            "h": num_h,
            "x": num_x,
            "ccx": num_ccx,
            "total_gates": total_gates,
            "success_rate": success_rate,
            "entropy": entropy,
            "unique_states": unique_states,
            "label": label
        }
    

.. code:: ipython3

    # Imports
    import pandas as pd
    import numpy as np
    from scipy.stats import entropy as shannon_entropy

.. code:: ipython3

    # --------------------------------------------------
    # Feature Extraction & Save: Clean DJ Circuits
    # --------------------------------------------------
    
    dj_clean_dataset = []
    
    # Loop through DJ clean circuits
    for i in range(10):
        name = f"dj_clean_{i}"
        circuit = dj_clean_circuits[name]
    
        # Transpile and run
        transpiled = transpile(circuit, backend=backend)
        sampler = norm_sampler(backend)
        job = sampler.run([transpiled])
        result = job.result()
    
        # Extract and collect features
        features = extract_features(name, circuit, result, label=0)
        dj_clean_dataset.append(features)
    
    # Convert to DataFrame
    df_dj_clean = pd.DataFrame(dj_clean_dataset)
    
    # Display and export
    print(df_dj_clean)
    df_dj_clean.to_csv("dj_clean_dataset.csv", index=False)
    


.. parsed-literal::

             name  depth  cx  h  x  ccx  total_gates  success_rate   entropy  \
    0  dj_clean_0      3   0  7  1    0           13     49.682617  1.113984   
    1  dj_clean_1      4   0  7  2    0           14     50.219727  1.103618   
    2  dj_clean_2      3   0  7  1    0           13     49.853516  1.111413   
    3  dj_clean_3      6   2  7  1    0           15     48.559570  1.390499   
    4  dj_clean_4      3   0  7  1    0           13     49.755859  1.102320   
    5  dj_clean_5      4   0  7  2    0           14     49.975586  1.107208   
    6  dj_clean_6      3   0  7  1    0           13     51.708984  1.102507   
    7  dj_clean_7      6   2  7  1    0           15     47.973633  1.409767   
    8  dj_clean_8      3   0  7  1    0           13     50.415039  1.102462   
    9  dj_clean_9      4   0  7  2    0           14     49.438477  1.106187   
    
       unique_states  label  
    0              8      0  
    1              8      0  
    2              8      0  
    3             10      0  
    4              8      0  
    5              9      0  
    6              8      0  
    7             14      0  
    8              8      0  
    9              8      0  
    

.. code:: ipython3

    # --------------------------------------------------
    # Malicious Deutsch-Jozsa Circuit Generation (Trojan Simulation)
    # --------------------------------------------------
    
    # To simulate tampering or quantum Trojans in quantum circuits, this section generates
    # 10 malicious Deutsch-Jozsa algorithm circuits. Each is subtly altered to mimic
    # intentional adversarial behavior, causing deviations in behavior or performance.
    
    # Attack Simulation Strategies:
    # 1. Gate Injection: Insert additional CX, X, or CCX gates at valid but non-functional positions
    # 2. Oracle Corruption: Modify oracle logic to produce incorrect classification
    # 3. Output Pollution: Flip output qubit or entangle with random gates post-Hadamard
    # 4. Redundancy Noise: Duplicate logic to increase decoherence susceptibility
    
    # Objectives:
    # - Introduce Trojan-like behavior in DJ circuits
    # - Create label = 1 dataset samples for supervised classification
    # - Ensure variation across different tampering strategies
    
    # All circuits in this section will be labeled as `1` (malicious) in the dataset.
    
    

.. code:: ipython3

    # --------------------------------------------------
    # Function: Generate Malicious Deutsch-Jozsa Circuit
    # --------------------------------------------------
    
    def generate_dj_malicious_variation(index):
        """
        Generate a tampered Deutsch-Jozsa quantum circuit to simulate Trojan attacks.
        Variations include oracle corruption, extra gates, or structural redundancy.
        """
        n = 3  # input qubits
        qc = QuantumCircuit(n + 1)
    
        # Step 1: Standard Initialization
        qc.x(n)
        qc.h(range(n + 1))
    
        # Step 2: Oracle Tampering
        if index % 3 == 0:
            # Inject X gates before CX (messes with input basis)
            qc.x([0, 1])
            qc.cx(0, n)
            qc.cx(1, n)
        elif index % 3 == 1:
            # Balanced oracle but entangled with wrong qubit
            qc.cx(2, n)
            qc.cx(1, 2)
        else:
            # Fake constant oracle but messes with output later
            qc.cx(0, n)
            qc.z(n)
    
        # Step 3: Standard Hadamards on input
        qc.h(range(n))
    
        # Step 4: Inject Trojan logic post-Hadamard
        if index % 2 == 0:
            qc.x(1)
        else:
            qc.cx(0, 2)
    
        # Step 5: Measurement
        qc.measure_all()
    
        return qc
    

.. code:: ipython3

    # --------------------------------------------------
    # Generate & Store All Malicious DJ Circuit Variants
    # --------------------------------------------------
    
    # Dictionary to hold malicious DJ circuits
    dj_malicious_circuits = {}
    
    # Generate 10 malicious variations
    for i in range(10):
        circuit = generate_dj_malicious_variation(i)
        dj_malicious_circuits[f"dj_malicious_{i}"] = circuit
    

.. code:: ipython3

    # --------------------------------------------------
    # Run & Save Malicious DJ Circuits on IBM Backend
    # --------------------------------------------------
    
    # Create folders for visualizations
    os.makedirs("dj_malicious_diagrams", exist_ok=True)
    os.makedirs("dj_malicious_histograms", exist_ok=True)
    
    results_summary_malicious = []
    
    for i in range(10):
        circuit_name = f"dj_malicious_{i}"
        circuit = dj_malicious_circuits[circuit_name]
    
        # Transpile and run on backend
        transpiled = transpile(circuit, backend=backend)
        sampler = norm_sampler(backend)
        job = sampler.run([transpiled])
        result = job.result()
    
        # Extract results
        counts = result[0].data.meas.get_counts()
        success_state = max(counts, key=counts.get)
        percentage = counts[success_state] / sum(counts.values()) * 100
    
        # Save circuit diagram
        circuit.draw("mpl", filename=f"dj_malicious_diagrams/{circuit_name}.png")
    
        # Plot and save histogram
        fig = plot_histogram(counts, title=f"{circuit_name} | {percentage:.2f}% success")
        fig.savefig(f"dj_malicious_histograms/{circuit_name}_histogram.png")
        plt.show()
    
        # Store result summary
        results_summary_malicious.append({
            "name": circuit_name,
            "success_state": success_state,
            "success_rate": percentage
        })
    
    # Display results
    for res in results_summary_malicious:
        print(f"{res['name']} | State: {res['success_state']} | Success Rate: {res['success_rate']:.2f}%")
    



.. image:: output_12_0.png



.. image:: output_12_1.png



.. image:: output_12_2.png



.. image:: output_12_3.png



.. image:: output_12_4.png



.. image:: output_12_5.png



.. image:: output_12_6.png



.. image:: output_12_7.png



.. image:: output_12_8.png



.. image:: output_12_9.png



.. image:: output_12_10.png



.. image:: output_12_11.png



.. image:: output_12_12.png



.. image:: output_12_13.png



.. image:: output_12_14.png



.. image:: output_12_15.png



.. image:: output_12_16.png



.. image:: output_12_17.png



.. image:: output_12_18.png



.. image:: output_12_19.png


.. parsed-literal::

    dj_malicious_0 | State: 0001 | Success Rate: 47.00%
    dj_malicious_1 | State: 1110 | Success Rate: 47.85%
    dj_malicious_2 | State: 1011 | Success Rate: 48.46%
    dj_malicious_3 | State: 1111 | Success Rate: 47.83%
    dj_malicious_4 | State: 0100 | Success Rate: 63.38%
    dj_malicious_5 | State: 1101 | Success Rate: 48.39%
    dj_malicious_6 | State: 1001 | Success Rate: 47.31%
    dj_malicious_7 | State: 1110 | Success Rate: 47.51%
    dj_malicious_8 | State: 1011 | Success Rate: 49.37%
    dj_malicious_9 | State: 1111 | Success Rate: 47.90%
    

.. code:: ipython3

    # --------------------------------------------------
    # Feature Extraction & Save: Malicious DJ Circuits
    # --------------------------------------------------
    
    dj_malicious_dataset = []
    
    # Loop through malicious DJ circuits
    for i in range(10):
        name = f"dj_malicious_{i}"
        circuit = dj_malicious_circuits[name]
    
        # Transpile and run
        transpiled = transpile(circuit, backend=backend)
        sampler = norm_sampler(backend)
        job = sampler.run([transpiled])
        result = job.result()
    
        # Extract and collect features
        features = extract_features(name, circuit, result, label=1)
        dj_malicious_dataset.append(features)
    
    # Convert to DataFrame
    df_dj_malicious = pd.DataFrame(dj_malicious_dataset)
    
    # Display and export
    print(df_dj_malicious)
    df_dj_malicious.to_csv("dj_malicious_dataset.csv", index=False)
    


.. parsed-literal::

                 name  depth  cx  h  x  ccx  total_gates  success_rate   entropy  \
    0  dj_malicious_0      7   2  7  4    0           18     47.949219  1.337219   
    1  dj_malicious_1      7   3  7  1    0           16     47.387695  1.449411   
    2  dj_malicious_2      5   1  7  2    0           16     48.852539  1.210870   
    3  dj_malicious_3      6   3  7  3    0           18     47.192383  1.519457   
    4  dj_malicious_4      7   2  7  2    0           16     61.596680  1.229426   
    5  dj_malicious_5      6   2  7  1    0           16     48.291016  1.316902   
    6  dj_malicious_6      7   2  7  4    0           18     48.413086  1.372343   
    7  dj_malicious_7      7   3  7  1    0           16     47.973633  1.528666   
    8  dj_malicious_8      5   1  7  2    0           16     48.779297  1.256007   
    9  dj_malicious_9      6   3  7  3    0           18     48.608398  1.452988   
    
       unique_states  label  
    0             11      1  
    1             13      1  
    2              8      1  
    3             15      1  
    4             10      1  
    5             11      1  
    6             10      1  
    7             13      1  
    8              9      1  
    9             15      1  
    

.. code:: ipython3

    # --------------------------------------------------
    # Merge Clean + Malicious DJ Datasets
    # --------------------------------------------------
    
    # Combine both datasets
    df_dj_combined = pd.concat([df_dj_clean, df_dj_malicious], ignore_index=True)
    
    # Save merged dataset
    df_dj_combined.to_csv("dj_full_dataset.csv", index=False)
    
    # Display preview
    print("Combined DJ Dataset (Clean + Malicious):")
    print(df_dj_combined.head(10))
    


.. parsed-literal::

    Combined DJ Dataset (Clean + Malicious):
             name  depth  cx  h  x  ccx  total_gates  success_rate   entropy  \
    0  dj_clean_0      3   0  7  1    0           13     49.682617  1.113984   
    1  dj_clean_1      4   0  7  2    0           14     50.219727  1.103618   
    2  dj_clean_2      3   0  7  1    0           13     49.853516  1.111413   
    3  dj_clean_3      6   2  7  1    0           15     48.559570  1.390499   
    4  dj_clean_4      3   0  7  1    0           13     49.755859  1.102320   
    5  dj_clean_5      4   0  7  2    0           14     49.975586  1.107208   
    6  dj_clean_6      3   0  7  1    0           13     51.708984  1.102507   
    7  dj_clean_7      6   2  7  1    0           15     47.973633  1.409767   
    8  dj_clean_8      3   0  7  1    0           13     50.415039  1.102462   
    9  dj_clean_9      4   0  7  2    0           14     49.438477  1.106187   
    
       unique_states  label  
    0              8      0  
    1              8      0  
    2              8      0  
    3             10      0  
    4              8      0  
    5              9      0  
    6              8      0  
    7             14      0  
    8              8      0  
    9              8      0  
    

.. code:: ipython3

    # --------------------------------------------------
    # Dataset Analysis: Class Balance & Feature Correlation
    # --------------------------------------------------
    
    # Create folder for plots
    os.makedirs("dj_analysis_plots", exist_ok=True)
    
    # ---------- CLASS BALANCE ----------
    label_counts = df_dj_combined['label'].value_counts().sort_index()
    label_df = pd.DataFrame({
        'Label': ['Clean', 'Malicious'],
        'Count': label_counts.values
    })
    
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Label', y='Count', data=label_df, palette='Set2')
    plt.title("Class Balance: Clean (0) vs Malicious (1)")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("dj_analysis_plots/dj_class_balance.png")
    plt.show()
    
    # ---------- FEATURE CORRELATION HEATMAP ----------
    
    numeric_df = df_dj_combined.drop(columns=["name"])
    
    # Drop columns with zero standard deviation
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    
    # Compute correlation on remaining features
    correlation = numeric_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor='gray',
        square=True,
        cbar_kws={'shrink': 0.8},
        annot_kws={"size": 9}
    )
    
    plt.title("Feature Correlation Heatmap – DJ Circuits (Filtered)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("dj_analysis_plots/dj_feature_correlation_heatmap_final.png", dpi=300)
    plt.show()
    
    


.. parsed-literal::

    C:\Users\zeesh\AppData\Local\Temp\ipykernel_34580\1068871846.py:17: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x='Label', y='Count', data=label_df, palette='Set2')
    


.. image:: output_15_1.png



.. image:: output_15_2.png


