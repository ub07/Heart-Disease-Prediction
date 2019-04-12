# Heart-Disease-Prediction

## Data Set Information:

  ### attribute documentation: 
  
      1. age: age in years
      
      2. sex: sex (1 = male; 0 = female)
      
      3.cp: chest pain type -- Value 1: typical angina -- Value 2: atypical angina -- Value 3: non-anginal pain -- Value 4: asymptomatic 
      
      4. trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
      
      5. chol: serum cholestoral in mg/dl
      
      6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
      
      7. restecg: resting electrocardiographic results -- Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
      
      8. thalach: maximum heart rate achieved 
      
      9.exang: exercise induced angina (1 = yes; 0 = no) 
      
      10.oldpeak: ST depression induced by exercise relative to rest 
      
      11.slope: the slope of the peak exercise ST segment 
      
      12.ca: number of major vessels (0-3) colored by flourosopy 
      
      13.tha: l3 = normal; 6 = fixed defect; 7 = reversable defect 
      
      14.target: 1 or 0 
      
I've coded this from scratch and the model was able to achieve 88 percent accuracy on the testing set that is considered to be quite good for a model coded from stratch

  ## It consists of different file namely:
  
    data: Which extract data and manipulate as required
    
    sigmoid: Contains sigmoid function
   
    intialize_with_zeroes: Create zero matrix
   
    optimize: Contiains optimization code
   
    predict: predict the target after optimization
   
    propagate: Cost and derivative are formed
    
    model: Consists of learning rates, num_iterations to predict the output

