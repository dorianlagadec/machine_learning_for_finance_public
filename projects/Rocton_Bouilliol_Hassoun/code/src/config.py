PARAMS_RSF = {
    "model__n_estimators": [100, 200],
    "model__min_samples_leaf": [5, 10, 20, 40],
    "model__min_samples_split": [10, 20],
    "model__max_features": ["sqrt"]
}

PARAMS_XGB = {

    "model__n_estimators": [500],
    "model__max_depth": [3, 4],
    "model__learning_rate": [0.01, 0.03],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.7, 1.0],
    "model__reg_lambda": [0.5, 1.0]
}