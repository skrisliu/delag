# DELAG
S. Liu, S. Wang and L. Zhang, "Daily Land Surface Temperature Reconstruction in Landsat Cross-Track Areas Using Deep Ensemble Learning With Uncertainty Quantification," in IEEE Transactions on Geoscience and Remote Sensing, doi: [10.1109/TGRS.2025.3643985](https://doi.org/10.1109/TGRS.2025.3643985).

Contact email: skrisliu@gmail.com 

This is a GitHub repo at [github.com/skrisliu/delag](https://github.com/skrisliu/delag)


## Code and data update ongoing: last updated 20251216

### Minimum Run [NYC]: Task #1 Clear-Sky With Real-World Cloud Patterns
```bash
git clone https://github.com/skrisliu/delag.git
cd delag
curl -L https://github.com/skrisliu/delag/releases/latest/download/nyc.7z -o nyc.7z
7z x ./nyc.7z -o./nyc
python nyc_task1_01.py # Run eATC, get 200 predictions.
python nyc_task1_02.py # Based on 200 predictions, Run GP. 
```


## Environment & Package Setup
Run these commands in sequence (terminal/Anaconda Prompt):

```bash
conda create --name py39c python=3.9    
conda activate py39c 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117   
pip install gpytorch==1.13  
pip install numpy==1.26.4   
```


## Preprocessing
(Folders are auto-created for each step)
1. Unzip all Landsat data for the target year. [auto-create `unzip` folder]
2. (Optional, for ldn & hkg) Reproject data to the same UTM zone.
3. Clip data to target UTM x/y bounds. [auto-create `clip` folder]
4. Reorder data files as `[YYYYMMDD]_[LC08/LC09]` (date_sensor). [auto-create `order` folder]
5. Build datacubes: `lsts`, `clearmasks`, `meanbands`, `era5lst`. [auto-create `datacube` folder]


## Ready-to-Use Training/Testing Dataset
### NYC
1. The NYC dataset is hosted via GitHub Releases:   [nyc.7z](https://github.com/skrisliu/delag/releases/latest/download/nyc.7z)  
2. After downloading, unzip it to the `./nyc` folder to run the scripts:  
```bash
curl -L https://github.com/skrisliu/delag/releases/latest/download/nyc.7z -o nyc.7z
7z x ./nyc.7z -o./nyc
```


## Experiments: three settings
We evaluate DELAG's performance across three experimental setups:

### Task #1: Clear-Sky Situations With Real-World Cloud Patterns
For a day with observed data, use another day's real-world cloud patterns
```bash
NYC: Predict index= 91, cloud index=155
LDN: Predict index=249, cloud index=169
HKG: Predict index=321, cloud index=177
```

#### NYC Task #1
```bash
python nyc_task1_01.py # Run eATC, get 200 predictions. 
python nyc_task1_02.py # Based on 200 predictions, Run GP. 
```

### Task #2: Heavily-Cloudy Situations
Reconstruction under heavily cloudy situations. 

### Task #3: Indirect Validation via Estimating Near-Surface Air Temperature
Indirectly evaluate performance through LST data's capability to estimate near-surface air temperature. 
1. The nyc.7z file should be downloaded and unzip to ./nyc
2. Download the full prediction data [~1.6GB]
3. Unzip to current folder
4. Run [nyc_task3_plot_airtemp.py](nyc_task3_plot_airtemp.py) to estimate air temperature from the predicted surface temperature.

```bash
# curl -L https://github.com/skrisliu/delag/releases/latest/download/nyc.7z -o nyc.7z
# 7z x ./nyc.7z -o./nyc
curl -L https://github.com/skrisliu/delag/releases/latest/download/nyc_pred_2023.zip -o nyc_pred_2023.zip
7z x nyc_pred_2023.zip
python nyc_task3_plot_airtemp.py
```




## Project Task Status
### Code cleaning in progress 
✅ Dependencies  
✅ Preprocessing Code  
✅ Task #1: Clear-Sky Situations With Real-World Cloud Patterns  
✅ Task #1: Data Ready  
❌ Task #2: Under Heavily-Cloudy Situations  
❌ Task #2: Data Ready  
✅ Task #3: Indirect Validation via Near-Surface Air Temperature  
✅ Task #3: Data Ready  


## The Paper

Liu, Shengjie, Siqin Wang, Lu Zhang. Daily Land Surface Temperature Reconstruction in Landsat Cross-Track Areas Using Deep Ensemble Learning With Uncertainty. Quantification. IEEE Transactions on Geoscience and Remote Sensing, 2025. [https://doi.org/10.1109/TGRS.2025.3643985](https://doi.org/10.1109/TGRS.2025.3643985)

```bibtex
@article{liu2025daily,
    author = {Liu, Shengjie and Wang, Siqin and Zhang, Lu},
    title = {Daily Land Surface Temperature Reconstruction in Landsat Cross-Track Areas Using Deep Ensemble Learning With Uncertainty Quantification},
    journal = {IEEE Transactions on Geoscience and Remote Sensing},
    year = {2025}, 
    doi = {10.1109/TGRS.2025.3643985}
}