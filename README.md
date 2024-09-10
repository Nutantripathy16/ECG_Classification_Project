# ECG_Classification_Project

ECG Data
The electrocardiogram (ECG) is a plot of voltage on the vertical axis against time on the 
horizontal axis. The electrodes are connected to a galvanometer that records a potential 
difference. The needle (or pen) of the ECG is deflected a given distance depending upon 
the voltage measured. Here for our project, we are using ECG Heartbeat Categorization 
Dataset from Kaggle [4]. Our dataset is composed of collection of heartbeat signals 
derived from a famous dataset in heartbeat classification, the MIT-BIH Arrhythmia
Dataset . The number of samples in both collections is large enough for training a deep 
neural network. Here we get the dataset already split into testing and training data that is
mitbih_test.csv and mitbih_train.csv 

The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal 
case and the cases affected by different arrhythmias. These signals are preprocessed 
and segmented, with each segment corresponding to a heartbeat.
For Arrhythmia Dataset
* Number of Samples: 109446
* Number of Categories: 5
* Sampling Frequency: 125Hz
* Data Source: Physionet's MIT-BIH Arrhythmia Dataset
* Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
  
This dataset consists of a series of CSV files. Each of these CSV files contain a matrix, 
with each row representing an example in that portion of the dataset. The final element of 
each row denotes the class to which that example belongs. The end of each column has 
figures 0-4, which specifies the class of the data, here 
* 0 = (N) Normal 
* 1 = (S)Atrial Premature
* 2 = (V) Premature ventricular contraction
* 3 = (F) fusion of ventricular and normal
* 4 = (Q) Fusion of paced and normal
