import joblib 
from  load_config import load_config
def create_pkl_file(model):
    
    config = load_config('config.yaml')
    model_path=config['data']['model']
    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}")