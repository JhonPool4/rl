# Package information
This package ...

# Requirements
- Python 3.8
- Opensim 4.3 
# Conda environemnt
create conda enviornment
<pre><code>conda env create -f conda_config/custom_env.yml </code></pre> 
activate conda environemnt
<pre><code>conda activate opensim-py38 </code></pre> 
install libraries to conda environment
<pre><code>pip install -r conda_config/</code></pre> 

# Configuration of Opensim 4.3
Go to ~/Opensim 4.3/sdk/Python
<pre><code>python setup_win_python38.py</code></pre> 
<pre><code>python -m pip install .</code></pre> 
Add path of the dynamic libraries
<pre><code>Path: C:\OpenSim 4.3\bin</code></pre> 
Add python libraries
<pre><code>PYTHONPATH: C:\OpenSim 4.3\sdk\Python</code></pre> 

# Test 
After activate conda environment
<pre><code>python main_customarm.py</code></pre> 

