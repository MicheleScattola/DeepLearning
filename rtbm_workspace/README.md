# Legacy RTBM Training

This workspace provides an isolated, legacy-compatible environment designed specifically to compile and execute the **Riemann-Theta Boltzmann Machine (RTBM)** framework (`RiemannAI/theta`).

---

## Prerequisites & System Packages

```bash
# 1. Add the legacy Python repository
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# 2. Install Python 3.8, its virtual environment module, and compiler headers
sudo apt install python3.8 python3.8-venv python3.8-dev python3.8-distutils -y
```

## Environment and compilation
```bash
# 1. Construct an isolated Python 3.8 virtual environment
python3.8 -m venv .venv

# 2. Activate the local environment bubble
source .venv/bin/activate

# 3. Upgrade pip internally
./.venv/bin/pip install --upgrade pip

# 4. Install legacy packages
./.venv/bin/pip install "numpy>=1.16.0,<1.20.0" "cython<3.0.0" cma matplotlib wheel scipy scikit-learn hyperopt

# 5. Compile 
./.venv/bin/pip install --no-build-isolation -e ./theta
```