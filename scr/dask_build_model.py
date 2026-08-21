from  load_config import load_config
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def build_model(preprocess):
    config = load_config('config.yaml')
    config_objective = config['data']['objective']
    config_eval_metric = config['data']['eval_metric']
    config_n_estimators = config['data']['n_estimators']
    config_max_depth = config['data']['max_depth']
    config_learning_rate = config['data']['learning_rate']
    config_subsample = config['data']['subsample']
    config_colsample_bytree = config['data']['colsample_bytree']
    config_n_jobs = config['data']['n_jobs']
    config_tree_method = config['data']['tree_method']
    config_reg_alpha = config['data']['reg_alpha']
    config_reg_lambda = config['data']['reg_lambda']
    config_random_state = config['data']['random_state']  
    

    xgb_model = XGBClassifier(
        objective=config_objective,
        eval_metric=config_eval_metric,
        n_estimators=config_n_estimators,
        max_depth=config_max_depth,
        learning_rate=config_learning_rate,
        subsample=config_subsample,
        colsample_bytree=config_colsample_bytree,
        n_jobs=config_n_jobs,
        tree_method=config_tree_method,
        reg_alpha=config_reg_alpha,
        reg_lambda=config_reg_lambda
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("imbalance", SMOTE(random_state=config_random_state)),
            ("xgb", xgb_model),
        ]
    )
    return model