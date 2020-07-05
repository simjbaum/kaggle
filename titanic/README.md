# kaggle titanic challenge


```python
import pandas as pd
import numpy as np
import plotly.express as px
import re
```

# load the data


```python
test = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/test.csv")
```


```python
test.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/train.csv")
```


```python
train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
df =  pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/gender_submission.csv")
```

# data processing - str to int or removing Nans


```python
test["gender"] = test.apply(lambda x: 0 if x["Sex"] == "male" else 1, axis = 1)
train["gender"] = train.apply(lambda x: 0 if x["Sex"] == "male" else 1, axis = 1)
```


```python
test[~test.Name.str.contains("Mr")]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>3</td>
      <td>Connolly, Miss. Kate</td>
      <td>female</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>330972</td>
      <td>7.6292</td>
      <td>NaN</td>
      <td>Q</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>910</td>
      <td>3</td>
      <td>Ilmakangas, Miss. Ida Livija</td>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
      <td>0</td>
      <td>STON/O2. 3101270</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>913</td>
      <td>3</td>
      <td>Olsen, Master. Artur Karl</td>
      <td>male</td>
      <td>9.0</td>
      <td>0</td>
      <td>1</td>
      <td>C 17368</td>
      <td>3.1708</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>918</td>
      <td>1</td>
      <td>Ostby, Miss. Helene Ragnhild</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>1</td>
      <td>113509</td>
      <td>61.9792</td>
      <td>B36</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>928</td>
      <td>3</td>
      <td>Roth, Miss. Sarah A</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>342712</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>409</th>
      <td>1301</td>
      <td>3</td>
      <td>Peacock, Miss. Treasteall</td>
      <td>female</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>SOTON/O.Q. 3101315</td>
      <td>13.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>410</th>
      <td>1302</td>
      <td>3</td>
      <td>Naughton, Miss. Hannah</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>365237</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>1</td>
    </tr>
    <tr>
      <th>412</th>
      <td>1304</td>
      <td>3</td>
      <td>Henriksson, Miss. Jenny Lovisa</td>
      <td>female</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>347086</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17758</td>
      <td>108.9000</td>
      <td>C105</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>106 rows × 12 columns</p>
</div>




```python
vec_test = []
for row in test["Embarked"]:
    if row == "Q":
        vec_test.append(0)
    if row == "S":
        vec_test.append(1)
    if row == "C":
        vec_test.append(2)
```


```python
test["Embarked_v1"] = vec_test
```


```python
vec_train = []
for row in train["Embarked"]:
    if row == "Q":
        vec_train.append(0)
    elif row == "S":
        vec_train.append(1)
    elif row == "C":
        vec_train.append(2)
    else:
        vec_train.append(3)
    
```


```python
train["Embarked_v1"] = vec_train
```


```python
# explore the age
```


```python
px.scatter(test, x = "Age")
```


<div>


            <div id="2b158905-4642-4ead-a11b-4191230a008e" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("2b158905-4642-4ead-a11b-4191230a008e")) {
                    Plotly.newPlot(
                        '2b158905-4642-4ead-a11b-4191230a008e',
                        [{"hovertemplate": "Age=%{x}<br>index=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "h", "showlegend": false, "type": "scatter", "x": [34.5, 47.0, 62.0, 27.0, 22.0, 14.0, 30.0, 26.0, 18.0, 21.0, null, 46.0, 23.0, 63.0, 47.0, 24.0, 35.0, 21.0, 27.0, 45.0, 55.0, 9.0, null, 21.0, 48.0, 50.0, 22.0, 22.5, 41.0, null, 50.0, 24.0, 33.0, null, 30.0, 18.5, null, 21.0, 25.0, null, 39.0, null, 41.0, 30.0, 45.0, 25.0, 45.0, null, 60.0, 36.0, 24.0, 27.0, 20.0, 28.0, null, 10.0, 35.0, 25.0, null, 36.0, 17.0, 32.0, 18.0, 22.0, 13.0, null, 18.0, 47.0, 31.0, 60.0, 24.0, 21.0, 29.0, 28.5, 35.0, 32.5, null, 55.0, 30.0, 24.0, 6.0, 67.0, 49.0, null, null, null, 27.0, 18.0, null, 2.0, 22.0, null, 27.0, null, 25.0, 25.0, 76.0, 29.0, 20.0, 33.0, 43.0, 27.0, null, 26.0, 16.0, 28.0, 21.0, null, null, 18.5, 41.0, null, 36.0, 18.5, 63.0, 18.0, null, 1.0, 36.0, 29.0, 12.0, null, 35.0, 28.0, null, 17.0, 22.0, null, 42.0, 24.0, 32.0, 53.0, null, null, 43.0, 24.0, 26.5, 26.0, 23.0, 40.0, 10.0, 33.0, 61.0, 28.0, 42.0, 31.0, null, 22.0, null, 30.0, 23.0, null, 60.5, 36.0, 13.0, 24.0, 29.0, 23.0, 42.0, 26.0, null, 7.0, 26.0, null, 41.0, 26.0, 48.0, 18.0, null, 22.0, null, 27.0, 23.0, null, 40.0, 15.0, 20.0, 54.0, 36.0, 64.0, 30.0, 37.0, 18.0, null, 27.0, 40.0, 21.0, 17.0, null, 40.0, 34.0, null, 11.5, 61.0, 8.0, 33.0, 6.0, 18.0, 23.0, null, null, 0.33, 47.0, 8.0, 25.0, null, 35.0, 24.0, 33.0, 25.0, 32.0, null, 17.0, 60.0, 38.0, 42.0, null, 57.0, 50.0, null, 30.0, 21.0, 22.0, 21.0, 53.0, null, 23.0, null, 40.5, 36.0, 14.0, 21.0, 21.0, null, 39.0, 20.0, 64.0, 20.0, 18.0, 48.0, 55.0, 45.0, 45.0, null, null, 41.0, 22.0, 42.0, 29.0, null, 0.92, 20.0, 27.0, 24.0, 32.5, null, null, 28.0, 19.0, 21.0, 36.5, 21.0, 29.0, 1.0, 30.0, null, null, null, null, 17.0, 46.0, null, 26.0, null, null, 20.0, 28.0, 40.0, 30.0, 22.0, 23.0, 0.75, null, 9.0, 2.0, 36.0, null, 24.0, null, null, null, 30.0, null, 53.0, 36.0, 26.0, 1.0, null, 30.0, 29.0, 32.0, null, 43.0, 24.0, null, 64.0, 30.0, 0.83, 55.0, 45.0, 18.0, 22.0, null, 37.0, 55.0, 17.0, 57.0, 19.0, 27.0, 22.0, 26.0, 25.0, 26.0, 33.0, 39.0, 23.0, 12.0, 46.0, 29.0, 21.0, 48.0, 39.0, null, 19.0, 27.0, 30.0, 32.0, 39.0, 25.0, null, 18.0, 32.0, null, 58.0, null, 16.0, 26.0, 38.0, 24.0, 31.0, 45.0, 25.0, 18.0, 49.0, 0.17, 50.0, 59.0, null, null, 30.0, 14.5, 24.0, 31.0, 27.0, 25.0, null, null, 22.0, 45.0, 29.0, 21.0, 31.0, 49.0, 44.0, 54.0, 45.0, 22.0, 21.0, 55.0, 5.0, null, 26.0, null, 19.0, null, 24.0, 24.0, 57.0, 21.0, 6.0, 23.0, 51.0, 13.0, 47.0, 29.0, 18.0, 24.0, 48.0, 22.0, 31.0, 30.0, 38.0, 22.0, 17.0, 43.0, 20.0, 23.0, 50.0, null, 3.0, null, 37.0, 28.0, null, 39.0, 38.5, null, null], "xaxis": "x", "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Age"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "index"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('2b158905-4642-4ead-a11b-4191230a008e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
# try first with giving age = 0 if nan
```


```python
test["Age"].fillna(0, inplace=True)
test["Age"] = test["Age"].astype('int64')

```


```python
train["Age"].fillna(0,inplace=True)
train["Age"] = train["Age"].astype('int64')
```


```python
# write Fare as numeric not float # and fill NA
```


```python
test["Fare"].fillna(0, inplace=True)
train["Fare"].fillna(0,inplace=True)
```


```python
train["Fare"] = train["Fare"].astype('int64')
test["Fare"] = test["Fare"].astype('int64')
```


```python
# fill all Nas with 0 for the beginning
```


```python
test.fillna(0, inplace=True)
train.fillna(0,inplace=True)
```


```python
# look at the tickets:
```


```python
#train["ticket_str"] = train.apply(lambda x: re.sub("\d+", "", x["Ticket"]), axis = 1)
```


```python
#pd.unique(train["ticket_str"] )
```


```python
# ok - I can come back later to this to make sense out of it
```


```python
# Fare
```


```python
px.box(train, "Pclass", "Fare")
```


<div>


            <div id="488b8358-e498-47c9-93a5-56343eec8554" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("488b8358-e498-47c9-93a5-56343eec8554")) {
                    Plotly.newPlot(
                        '488b8358-e498-47c9-93a5-56343eec8554',
                        [{"alignmentgroup": "True", "hovertemplate": "Pclass=%{x}<br>Fare=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa"}, "name": "", "notched": false, "offsetgroup": "", "orientation": "v", "showlegend": false, "type": "box", "x": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, 1, 3, 3, 3, 1, 3, 3, 1, 1, 3, 2, 1, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 1, 1, 2, 3, 2, 3, 3, 1, 1, 3, 1, 3, 2, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 2, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 1, 1, 2, 2, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 2, 1, 3, 2, 3, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 3, 1, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 3, 1, 3, 3, 1, 2, 3, 3, 2, 3, 2, 3, 1, 3, 1, 3, 3, 2, 2, 3, 2, 1, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 2, 3, 1, 3, 2, 1, 2, 3, 2, 3, 3, 1, 3, 2, 3, 2, 3, 1, 3, 2, 3, 2, 3, 2, 2, 2, 2, 3, 3, 2, 3, 3, 1, 3, 2, 1, 2, 3, 3, 1, 3, 3, 3, 1, 1, 1, 2, 3, 3, 1, 1, 3, 2, 3, 3, 1, 1, 1, 3, 2, 1, 3, 1, 3, 2, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 2, 3, 1, 1, 2, 3, 3, 1, 3, 1, 1, 1, 3, 3, 3, 2, 3, 1, 1, 1, 2, 1, 1, 1, 2, 3, 2, 3, 2, 2, 1, 1, 3, 3, 2, 2, 3, 1, 3, 2, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 2, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 3, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 2, 3, 2, 3, 2, 1, 3, 3, 1, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 1, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 1, 3, 2, 3, 1, 1, 3, 2, 1, 2, 2, 3, 3, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 1, 3, 3, 1, 1, 2, 3, 1, 3, 1, 2, 3, 3, 2, 1, 3, 3, 3, 3, 2, 2, 3, 1, 2, 3, 3, 3, 3, 2, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 1, 1, 2, 1, 3, 3, 3, 3, 1, 1, 3, 1, 2, 3, 2, 3, 1, 3, 3, 1, 3, 3, 2, 1, 3, 2, 2, 3, 3, 3, 3, 2, 1, 1, 3, 1, 1, 3, 3, 2, 1, 1, 2, 2, 3, 2, 1, 2, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 3, 3, 3, 2, 1, 3, 3, 2, 1, 2, 1, 3, 1, 2, 1, 3, 3, 3, 1, 3, 3, 2, 3, 2, 3, 3, 1, 2, 3, 1, 3, 1, 3, 3, 1, 2, 1, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 1, 3, 3, 3, 1, 2, 1, 3, 3, 1, 3, 1, 1, 3, 2, 3, 2, 3, 3, 3, 1, 3, 3, 3, 1, 3, 1, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2, 1, 1, 3, 1, 3, 3, 2, 2, 3, 3, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3, 1, 3, 1, 3, 3, 2, 2, 3, 3, 3, 1, 1, 3, 3, 3, 1, 2, 3, 3, 1, 3, 1, 1, 3, 3, 3, 2, 2, 1, 1, 3, 1, 1, 1, 3, 2, 3, 1, 2, 3, 3, 2, 3, 2, 2, 1, 3, 2, 3, 2, 3, 1, 3, 2, 2, 2, 3, 3, 1, 3, 3, 1, 1, 1, 3, 3, 1, 3, 2, 1, 3, 2, 3, 3, 3, 2, 2, 3, 2, 3, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 1, 3, 2, 1, 3, 3, 3, 2, 2, 1, 3, 3, 3, 1, 3, 2, 1, 3, 3, 2, 3, 3, 1, 3, 2, 3, 3, 1, 3, 1, 3, 3, 3, 3, 2, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 1, 3, 2, 1, 3, 3, 3, 3, 3, 2, 1, 3, 3, 3, 1, 2, 3, 1, 1, 3, 3, 3, 2, 1, 3, 2, 2, 2, 1, 3, 3, 3, 1, 1, 3, 2, 3, 3, 3, 3, 1, 2, 3, 3, 2, 3, 3, 2, 1, 3, 1, 3], "x0": " ", "xaxis": "x", "y": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05, 31.275, 7.8542, 16.0, 29.125, 13.0, 18.0, 7.225, 26.0, 13.0, 8.0292, 35.5, 21.075, 31.3875, 7.225, 263.0, 7.8792, 7.8958, 27.7208, 146.5208, 7.75, 10.5, 82.1708, 52.0, 7.2292, 8.05, 18.0, 11.2417, 9.475, 21.0, 7.8958, 41.5792, 7.8792, 8.05, 15.5, 7.75, 21.6792, 17.8, 39.6875, 7.8, 76.7292, 26.0, 61.9792, 35.5, 10.5, 7.2292, 27.75, 46.9, 7.2292, 80.0, 83.475, 27.9, 27.7208, 15.2458, 10.5, 8.1583, 7.925, 8.6625, 10.5, 46.9, 73.5, 14.4542, 56.4958, 7.65, 7.8958, 8.05, 29.0, 12.475, 9.0, 9.5, 7.7875, 47.1, 10.5, 15.85, 34.375, 8.05, 263.0, 8.05, 8.05, 7.8542, 61.175, 20.575, 7.25, 8.05, 34.6542, 63.3583, 23.0, 26.0, 7.8958, 7.8958, 77.2875, 8.6542, 7.925, 7.8958, 7.65, 7.775, 7.8958, 24.15, 52.0, 14.4542, 8.05, 9.825, 14.4583, 7.925, 7.75, 21.0, 247.5208, 31.275, 73.5, 8.05, 30.0708, 13.0, 77.2875, 11.2417, 7.75, 7.1417, 22.3583, 6.975, 7.8958, 7.05, 14.5, 26.0, 13.0, 15.0458, 26.2833, 53.1, 9.2167, 79.2, 15.2458, 7.75, 15.85, 6.75, 11.5, 36.75, 7.7958, 34.375, 26.0, 13.0, 12.525, 66.6, 8.05, 14.5, 7.3125, 61.3792, 7.7333, 8.05, 8.6625, 69.55, 16.1, 15.75, 7.775, 8.6625, 39.6875, 20.525, 55.0, 27.9, 25.925, 56.4958, 33.5, 29.125, 11.1333, 7.925, 30.6958, 7.8542, 25.4667, 28.7125, 13.0, 0.0, 69.55, 15.05, 31.3875, 39.0, 22.025, 50.0, 15.5, 26.55, 15.5, 7.8958, 13.0, 13.0, 7.8542, 26.0, 27.7208, 146.5208, 7.75, 8.4042, 7.75, 13.0, 9.5, 69.55, 6.4958, 7.225, 8.05, 10.4625, 15.85, 18.7875, 7.75, 31.0, 7.05, 21.0, 7.25, 13.0, 7.75, 113.275, 7.925, 27.0, 76.2917, 10.5, 8.05, 13.0, 8.05, 7.8958, 90.0, 9.35, 10.5, 7.25, 13.0, 25.4667, 83.475, 7.775, 13.5, 31.3875, 10.5, 7.55, 26.0, 26.25, 10.5, 12.275, 14.4542, 15.5, 10.5, 7.125, 7.225, 90.0, 7.775, 14.5, 52.5542, 26.0, 7.25, 10.4625, 26.55, 16.1, 20.2125, 15.2458, 79.2, 86.5, 512.3292, 26.0, 7.75, 31.3875, 79.65, 0.0, 7.75, 10.5, 39.6875, 7.775, 153.4625, 135.6333, 31.0, 0.0, 19.5, 29.7, 7.75, 77.9583, 7.75, 0.0, 29.125, 20.25, 7.75, 7.8542, 9.5, 8.05, 26.0, 8.6625, 9.5, 7.8958, 13.0, 7.75, 78.85, 91.0792, 12.875, 8.85, 7.8958, 27.7208, 7.2292, 151.55, 30.5, 247.5208, 7.75, 23.25, 0.0, 12.35, 8.05, 151.55, 110.8833, 108.9, 24.0, 56.9292, 83.1583, 262.375, 26.0, 7.8958, 26.25, 7.8542, 26.0, 14.0, 164.8667, 134.5, 7.25, 7.8958, 12.35, 29.0, 69.55, 135.6333, 6.2375, 13.0, 20.525, 57.9792, 23.25, 28.5, 153.4625, 18.0, 133.65, 7.8958, 66.6, 134.5, 8.05, 35.5, 26.0, 263.0, 13.0, 13.0, 13.0, 13.0, 13.0, 16.1, 15.9, 8.6625, 9.225, 35.0, 7.2292, 17.8, 7.225, 9.5, 55.0, 13.0, 7.8792, 7.8792, 27.9, 27.7208, 14.4542, 7.05, 15.5, 7.25, 75.25, 7.2292, 7.75, 69.3, 55.4417, 6.4958, 8.05, 135.6333, 21.075, 82.1708, 7.25, 211.5, 4.0125, 7.775, 227.525, 15.7417, 7.925, 52.0, 7.8958, 73.5, 46.9, 13.0, 7.7292, 12.0, 120.0, 7.7958, 7.925, 113.275, 16.7, 7.7958, 7.8542, 26.0, 10.5, 12.65, 7.925, 8.05, 9.825, 15.85, 8.6625, 21.0, 7.75, 18.75, 7.775, 25.4667, 7.8958, 6.8583, 90.0, 0.0, 7.925, 8.05, 32.5, 13.0, 13.0, 24.15, 7.8958, 7.7333, 7.875, 14.4, 20.2125, 7.25, 26.0, 26.0, 7.75, 8.05, 26.55, 16.1, 26.0, 7.125, 55.9, 120.0, 34.375, 18.75, 263.0, 10.5, 26.25, 9.5, 7.775, 13.0, 8.1125, 81.8583, 19.5, 26.55, 19.2583, 30.5, 27.75, 19.9667, 27.75, 89.1042, 8.05, 7.8958, 26.55, 51.8625, 10.5, 7.75, 26.55, 8.05, 38.5, 13.0, 8.05, 7.05, 0.0, 26.55, 7.725, 19.2583, 7.25, 8.6625, 27.75, 13.7917, 9.8375, 52.0, 21.0, 7.0458, 7.5208, 12.2875, 46.9, 0.0, 8.05, 9.5875, 91.0792, 25.4667, 90.0, 29.7, 8.05, 15.9, 19.9667, 7.25, 30.5, 49.5042, 8.05, 14.4583, 78.2667, 15.1, 151.55, 7.7958, 8.6625, 7.75, 7.6292, 9.5875, 86.5, 108.9, 26.0, 26.55, 22.525, 56.4958, 7.75, 8.05, 26.2875, 59.4, 7.4958, 34.0208, 10.5, 24.15, 26.0, 7.8958, 93.5, 7.8958, 7.225, 57.9792, 7.2292, 7.75, 10.5, 221.7792, 7.925, 11.5, 26.0, 7.2292, 7.2292, 22.3583, 8.6625, 26.25, 26.55, 106.425, 14.5, 49.5, 71.0, 31.275, 31.275, 26.0, 106.425, 26.0, 26.0, 13.8625, 20.525, 36.75, 110.8833, 26.0, 7.8292, 7.225, 7.775, 26.55, 39.6, 227.525, 79.65, 17.4, 7.75, 7.8958, 13.5, 8.05, 8.05, 24.15, 7.8958, 21.075, 7.2292, 7.8542, 10.5, 51.4792, 26.3875, 7.75, 8.05, 14.5, 13.0, 55.9, 14.4583, 7.925, 30.0, 110.8833, 26.0, 40.125, 8.7125, 79.65, 15.0, 79.2, 8.05, 8.05, 7.125, 78.2667, 7.25, 7.75, 26.0, 24.15, 33.0, 0.0, 7.225, 56.9292, 27.0, 7.8958, 42.4, 8.05, 26.55, 15.55, 7.8958, 30.5, 41.5792, 153.4625, 31.275, 7.05, 15.5, 7.75, 8.05, 65.0, 14.4, 16.1, 39.0, 10.5, 14.4542, 52.5542, 15.7417, 7.8542, 16.1, 32.3208, 12.35, 77.9583, 7.8958, 7.7333, 30.0, 7.0542, 30.5, 0.0, 27.9, 13.0, 7.925, 26.25, 39.6875, 16.1, 7.8542, 69.3, 27.9, 56.4958, 19.2583, 76.7292, 7.8958, 35.5, 7.55, 7.55, 7.8958, 23.0, 8.4333, 7.8292, 6.75, 73.5, 7.8958, 15.5, 13.0, 113.275, 133.65, 7.225, 25.5875, 7.4958, 7.925, 73.5, 13.0, 7.775, 8.05, 52.0, 39.0, 52.0, 10.5, 13.0, 0.0, 7.775, 8.05, 9.8417, 46.9, 512.3292, 8.1375, 76.7292, 9.225, 46.9, 39.0, 41.5792, 39.6875, 10.1708, 7.7958, 211.3375, 57.0, 13.4167, 56.4958, 7.225, 26.55, 13.5, 8.05, 7.7333, 110.8833, 7.65, 227.525, 26.2875, 14.4542, 7.7417, 7.8542, 26.0, 13.5, 26.2875, 151.55, 15.2458, 49.5042, 26.55, 52.0, 9.4833, 13.0, 7.65, 227.525, 10.5, 15.5, 7.775, 33.0, 7.0542, 13.0, 13.0, 53.1, 8.6625, 21.0, 7.7375, 26.0, 7.925, 211.3375, 18.7875, 0.0, 13.0, 13.0, 16.1, 34.375, 512.3292, 7.8958, 7.8958, 30.0, 78.85, 262.375, 16.1, 7.925, 71.0, 20.25, 13.0, 53.1, 7.75, 23.0, 12.475, 9.5, 7.8958, 65.0, 14.5, 7.7958, 11.5, 8.05, 86.5, 14.5, 7.125, 7.2292, 120.0, 7.775, 77.9583, 39.6, 7.75, 24.15, 8.3625, 9.5, 7.8542, 10.5, 7.225, 23.0, 7.75, 7.75, 12.475, 7.7375, 211.3375, 7.2292, 57.0, 30.0, 23.45, 7.05, 7.25, 7.4958, 29.125, 20.575, 79.2, 7.75, 26.0, 69.55, 30.6958, 7.8958, 13.0, 25.9292, 8.6833, 7.2292, 24.15, 13.0, 26.25, 120.0, 8.5167, 6.975, 7.775, 0.0, 7.775, 13.0, 53.1, 7.8875, 24.15, 10.5, 31.275, 8.05, 0.0, 7.925, 37.0042, 6.45, 27.9, 93.5, 8.6625, 0.0, 12.475, 39.6875, 6.95, 56.4958, 37.0042, 7.75, 80.0, 14.4542, 18.75, 7.2292, 7.8542, 8.3, 83.1583, 8.6625, 8.05, 56.4958, 29.7, 7.925, 10.5, 31.0, 6.4375, 8.6625, 7.55, 69.55, 7.8958, 33.0, 89.1042, 31.275, 7.775, 15.2458, 39.4, 26.0, 9.35, 164.8667, 26.55, 19.2583, 7.2292, 14.1083, 11.5, 25.9292, 69.55, 13.0, 13.0, 13.8583, 50.4958, 9.5, 11.1333, 7.8958, 52.5542, 5.0, 9.0, 24.0, 7.225, 9.8458, 7.8958, 7.8958, 83.1583, 26.0, 7.8958, 10.5167, 10.5, 7.05, 29.125, 13.0, 30.0, 23.45, 30.0, 7.75], "y0": " ", "yaxis": "y"}],
                        {"boxmode": "group", "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Pclass"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "Fare"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('488b8358-e498-47c9-93a5-56343eec8554');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
# check cabins
```


```python
pd.unique(train["Cabin"])
```




    array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
           'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
           'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
           'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
           'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
           'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
           'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
           'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
           'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44',
           'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14',
           'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38',
           'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68',
           'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48',
           'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
           'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
           'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
           'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
           'C148'], dtype=object)



# ok train a simple SVM model on the data:


```python
#remove PassengerId, Name, Sex (we have in a another variable), Ticket, Cabin, Embarked
```


```python
train.drop(columns = ["Name", "PassengerId", "Sex", "Ticket", "Cabin", "Embarked"], inplace=True)
test.drop(columns = ["Name", "PassengerId", "Sex", "Ticket", "Cabin", "Embarked"], inplace=True)
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>gender</th>
      <th>Embarked_v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>




```python
train.iloc[:,1:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>gender</th>
      <th>Embarked_v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>




```python
train.iloc[:,:1]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-226-8b77113bd72f> in <module>
    ----> 1 pd.Series(train.iloc[:,:1])
    

    ~/Library/Python/3.7/lib/python/site-packages/pandas/core/series.py in __init__(self, data, index, dtype, name, copy, fastpath)
        200             name = ibase.maybe_extract_name(name, data, type(self))
        201 
    --> 202             if is_empty_data(data) and dtype is None:
        203                 # gh-17261
        204                 warnings.warn(


    ~/Library/Python/3.7/lib/python/site-packages/pandas/core/construction.py in is_empty_data(data)
        584     is_none = data is None
        585     is_list_like_without_dtype = is_list_like(data) and not hasattr(data, "dtype")
    --> 586     is_simple_empty = is_list_like_without_dtype and not data
        587     return is_none or is_simple_empty
        588 


    ~/Library/Python/3.7/lib/python/site-packages/pandas/core/generic.py in __nonzero__(self)
       1477     def __nonzero__(self):
       1478         raise ValueError(
    -> 1479             f"The truth value of a {type(self).__name__} is ambiguous. "
       1480             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
       1481         )


    ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().



```python
from sklearn import svm
```


```python
X = np.array(train.iloc[:,1:])
y = np.array(train.iloc[:,:1]).squeeze()
```


```python
x = np.array(test)
```


```python
clf = svm.SVC()
clf.fit(X, y)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
res = clf.predict(x)
```


```python
test = pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/test.csv")
```


```python

```




    <PandasArray>
    [ 892,  893,  894,  895,  896,  897,  898,  899,  900,  901,
     ...
     1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309]
    Length: 418, dtype: int64




```python
to_submit = pd.DataFrame({"PassengerId":pd.array(test["PassengerId"]), 
             "Survived":res})
```


```python
!mkdir "Documents/Bioinformatics/kaggle/titanic/submission"
```


```python
to_submit.to_csv("Documents/Bioinformatics/kaggle/titanic/submission/05072020_titanic_SVM.csv", index=False)
```

### example submission data format


```python
pd.read_csv("Documents/Bioinformatics/kaggle/titanic/submission/05072020_titanic_SVM.csv")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 2 columns</p>
</div>




```python
pd.read_csv("Documents/Bioinformatics/kaggle/titanic/data/gender_submission.csv")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>1</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>418 rows × 2 columns</p>
</div>




```python

```
