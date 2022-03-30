# qkernel
Reproduction of [Machine learning of high dimensional data on a noisy quantum processor](https://arxiv.org/abs/2101.09581).

## Run
```shell
python main.py --n_qubits=5 --n_samples=1000 --qgamma=0.1 --pq=0.1 --rgamma=0.012 --cv=4
```

## Discrepancy & Question
- Number of features

  67 VS. 53
    
  53 features: 

  <div>
  <table border="1" class="dataframe">
    <thead>
      <tr align=right>
        <th></th>
        <th>feature</th>
        <th></th>
        <th>feature</th>
        <th></th>
        <th>feature</th>
        <th></th>
        <th>feature</th>
      </tr>
    </thead>
    <tbody>
      <tr align=right>
        <th>1</th>
        <td>n_measure</td>
        <th>16</th>
        <td>sum_flux_err2</td>
        <th>31</th>
        <td>fou2_0</td>
        <th>46</th>
        <td>fou1_4</td>
      </tr>
      <tr align=right>
        <th>2</th>
        <td>min_flux</td>
        <th>17</th>
        <td>skew_flux_err2</td>
        <th>32</th>
        <td>kur_0</td>
        <th>47</th>
        <td>fou2_4</td>
      </tr>
      <tr align=right>
        <th>3</th>
        <td>max_flux</td>
        <th>18</th>
        <td>mean_interval</td>
        <th>33</th>
        <td>skew_0</td>
        <th>48</th>
        <td>kur_4</td>
      </tr>
      <tr align=right>
        <th>4</th>
        <td>mean_flux</td>
        <th>19</th>
        <td>max_interval</td>
        <th>34</th>
        <td>fou1_1</td>
        <th>49</th>
        <td>skew_4</td>
      </tr>
      <tr align=right>
        <th>5</th>
        <td>med_flux</td>
        <th>20</th>
        <td>hostgal_specz</td>
        <th>35</th>
        <td>fou2_1</td>
        <th>50</th>
        <td>fou1_5</td>
      </tr>
       <tr align=right>
        <th>6</th>
        <td>std_flux</td>
        <th>21</th>
        <td>hostgal_photoz</td>
        <th>36</th>
        <td>kur_1</td> 
        <th>51</th>
        <td>fou2_5</td>
      </tr>
       <tr align=right>
        <th>7</th>
        <td>skew_flux</td>
        <th>22</th>
        <td>hostgal_photoz_err</td>
        <th>37</th>
        <td>skew_1</td> 
        <th>52</th>
        <td>kur_5</td>
      </tr>
        <tr align=right>
        <th>8</th>
        <td>min_flux_err</td>
        <th>23</th>
        <td>ra</td>
        <th>38</th>
        <td>fou1_2</td>
        <th>53</th>
        <td>skew_5</td>
      </tr>
      <tr align=right>
        <th>9</th>
        <td>max_flux_err</td>
        <th>24</th>
        <td>decl</td>
        <th>39</th>
        <td>fou2_2</td> 
      </tr>
      <tr align=right>
        <th>10</th>
        <td>mean_flux_err</td>
        <th>25</th>
        <td>gal_l</td>
        <th>40</th>
        <td>kur_2</td> 
      </tr>
      <tr align=right>
        <th>11</th>
        <td>med_flux_err</td>
        <th>26</th>
        <td>gal_b</td>
        <th>41</th>
        <td>skew_2</td> 
      </tr>
      <tr align=right>
        <th>12</th>
        <td>std_flux_err</td>
        <th>27</th>
        <td>ddf</td>
        <th>42</th>
        <td>fou1_3</td> 
      </tr>
       <tr align=right>
        <th>13</th>
        <td>skew_flux_err</td>
        <th>28</th>
        <td>distmod</td>
        <th>43</th>
        <td>fou2_3</td>
      </tr>
      <tr align=right>
        <th>14</th>
        <td>sum_flux_err_ratio</td>
        <th>29</th>
        <td>mwebv</td>
        <th>44</th>
        <td>kur_3</td>
      </tr>
      <tr align=right>
        <th>15</th>
        <td>skew_flux_err_ratio</td>
        <th>30</th>
        <td>fou1_0</td>
        <th>45</th>
        <td>skew_3</td>
      </tr>
    </tbody>
  </table>
  </div>

- Fourier coefficient

    Omit imaginary part?
- Logscale
```python
logscale_features = ['min_flux', 'max_flux', 'mean_flux', 'med_flux', 'std_flux', 'skew_flux',\
                    'min_flux_err', 'max_flux_err', 'mean_flux_err', 'med_flux_err', 'std_flux_err',\
                    'skew_flux_err', 'sum_flux_err_ratio', 'skew_flux_err_ratio', 'sum_flux_err2',\
                    'skew_flux_err2', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err']
```
- Robust scaler

    transformation in paper:
 $$x_k'=\pi (\frac{x_k-P_1}{P_{99}-P_1})-\frac{\pi}{2}$$
     different from [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
