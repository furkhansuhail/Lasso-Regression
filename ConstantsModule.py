Dataset_Link = "https://github.com/furkhansuhail/ProjectData/raw/refs/heads/main/LassoRegression/Experience-Salary.csv"
import statsmodels.api as sm
from dataclasses import dataclass
from pathlib import Path
import urllib.request as request

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
