## Installation for developers

Note, original requirements.yml fails.

```bash
git clone https://github.com/delftcrowd/SECA.git
cd SECA
conda env create --file conda/environment.yml
conda activate seca
pip install -e .[dev]
```
