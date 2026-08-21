from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, TargetEncoder
from  load_config import load_config

def build_preprocessor(df,num_col, cat_col):
   config = load_config('config.yaml')
   config_random_state = config['data']['random_state']
   n_splits = config['data']['n_splits']     
   
   transform_num = Pipeline(steps=[('scale_num',PowerTransformer(method='yeo-johnson'))])
   transform_cat = Pipeline(steps=[('scale_cat',TargetEncoder(random_state=config['data']['random_state'], cv=config['data']['n_splits']))])
   
   preprocess = ColumnTransformer(transformers=[('nums', transform_num,num_col),
																																																('cat',transform_cat,cat_col)
																																																], remainder='drop', verbose=True)
   return preprocess