# Smoother
This repository is the implementation for the paper `Simplified Kalman smoother and ensemble Kalman smoother for improving ocean forecasts and reanalyses`. We made slight modifications to the original [dapper](https://github.com/nansencenter/DAPPER) templates for our own purpose. Our modifications include: 
- Recording the ensemble trajectory and error covariance of the model
- An extended fix-lag smoother
- An incremental 3DVar with incremental analysis update
- Simplified exteneded and ensemble Kalman smoother
- Adaptation for time-dependent observation dimension

Refer to [`extks.py`](extks.py) and [`enks.py`](enks.py) for running experiments. Contact the correspondence author [Bo Dong](mailto:bo.dong@reading.ac.uk) if you have any questions.
