def get_data(data_path):
  import pandas as pd
  from sklearn.preprocessing import LabelBinarizer

  data = pd.read_csv(data_path)
  
  X = data.drop('y', axis=1)
  y = data.y.apply(lambda x: 1 if x == 'yes' else 0)

  return X, y