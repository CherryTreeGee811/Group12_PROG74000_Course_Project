def evaluate_all():
    """Load saved models, evaluate on test set, and generate charts."""
    cfg = _load_config()
    save_dir = os.path.join(_PROJECT_ROOT, cfg["output"]["model_save_dir"])

    # --- Load test set and feature columns ---
    test_df = pd.read_csv(
        os.path.join(save_dir, "test_set.csv"),
        index_col="Date", parse_dates=True,
    )
    feature_cols = _load_artifact("feature_columns.pkl", save_dir)

    X_test = test_df[feature_cols].values
    y_test_classification = test_df["Direction"].values
    y_test_regression = test_df["Next_Close"].values
    y_test_pct = test_df["Pct_Change"].values
    test_close = test_df["Close"].values

    print(f"[evaluate] Test set: {len(test_df)} rows, {len(feature_cols)} features")

    all_metrics = {}

    # =================================================================
    # Model 1 — Logistic Regression (skip if artifacts missing)
    # =================================================================
    print("\n" + "=" * 60)
    print("[evaluate] Model 1 — Logistic Regression")
    print("=" * 60)

    logistic_model_path = os.path.join(save_dir, "logistic_regression.pkl")
    logistic_scaler_path = os.path.join(save_dir, "logistic_scaler.pkl")
    if not os.path.exists(logistic_model_path) or not os.path.exists(logistic_scaler_path):
        print("  WARNING: Logistic regression artifacts not found. Skipping evaluation.")
    else:
        lr_model = _load_artifact("logistic_regression.pkl", save_dir)
        lr_scaler = _load_artifact("logistic_scaler.pkl", save_dir)
        X_test_lr = lr_scaler.transform(X_test)
        y_pred_lr = lr_model.predict(X_test_lr)
        all_metrics["Logistic Regression"] = _classification_metrics(
            y_test_classification, y_pred_lr, "Logistic Regression")

    # =================================================================
    # Model 2 — Polynomial Regression (skip if artifacts missing)
    # =================================================================
    print("=" * 60)
    print("[evaluate] Model 2 — Polynomial Regression")
    print("=" * 60)

    poly_model_path = os.path.join(save_dir, "polynomial_regression.pkl")
    poly_scaler_path = os.path.join(save_dir, "polynomial_scaler.pkl")
    poly_reg_metrics = None
    if not os.path.exists(poly_model_path) or not os.path.exists(poly_scaler_path):
        print("  WARNING: Polynomial regression artifacts not found. Skipping evaluation.")
    else:
        poly_model = _load_artifact("polynomial_regression.pkl", save_dir)
        poly_scaler = _load_artifact("polynomial_scaler.pkl", save_dir)
        X_test_poly = poly_scaler.transform(X_test)
        y_pred_poly_pct = poly_model.predict(X_test_poly)
        y_pred_poly_regression = test_close * (1 + y_pred_poly_pct / 100)
        poly_reg_metrics = _regression_metrics(
            y_test_regression, y_pred_poly_regression, "Polynomial Regression")
        

    # =================================================================
    # Model 3 — Linear Regression Regressor (skip if artifacts missing)
    # =================================================================
    print("=" * 60)
    print("[evaluate] Model 3 — Linear Regression")
    print("=" * 60)

    lr_reg_model_path  = os.path.join(save_dir, "linear_regression.pkl")
    lr_reg_scaler_path = os.path.join(save_dir, "linear_regression_scaler.pkl")
    lin_reg_metrics = None

    if not os.path.exists(lr_reg_model_path) or not os.path.exists(lr_reg_scaler_path):
        print("  WARNING: Linear Regression artifacts not found. Skipping evaluation.")
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        lr_reg_model  = _load_artifact("linear_regression.pkl",        save_dir)
        lr_reg_scaler = _load_artifact("linear_regression_scaler.pkl", save_dir)

        X_test_lr_reg     = lr_reg_scaler.transform(X_test)
        y_pred_lr_reg_pct = lr_reg_model.predict(X_test_lr_reg)

        # Convert pct change → dollar price (same as XGBoost regressor)
        y_pred_lr_reg = test_close * (1 + y_pred_lr_reg_pct / 100)

        lr_reg_metrics = {
            "MAE":  round(mean_absolute_error(y_test_regression, y_pred_lr_reg), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test_regression, y_pred_lr_reg)), 4),
            "R2":   round(r2_score(y_test_regression, y_pred_lr_reg), 4),
        }
        print("  Linear Regression Regression metrics:")
        for k, v in lr_reg_metrics.items():
            print(f"    {k}: {v}")


    # =================================================================
    # Model 4 — XGBoost Classifier + Regressor
    # =================================================================
    print("=" * 60)
    print("[evaluate] Model 4 — XGBoost")
    print("=" * 60)

    xgb_classifier = _load_artifact("xgboost_classifier.pkl", save_dir)
    xgb_regressor = _load_artifact("xgboost_regressor.pkl", save_dir)
    xgb_scaler = _load_artifact("xgboost_scaler.pkl", save_dir)

    X_test_xgb = xgb_scaler.transform(X_test)

    # Classification
    y_pred_xgb_classification = xgb_classifier.predict(X_test_xgb)
    all_metrics["XGBoost Classifier"] = _classification_metrics(
        y_test_classification, y_pred_xgb_classification, "XGBoost Classifier")

    # Regression — model predicts pct change; convert back to prices
    y_pred_pct = xgb_regressor.predict(X_test_xgb)
    y_pred_xgb_regressor = test_close * (1 + y_pred_pct / 100)
    reg_metrics = _regression_metrics(
        y_test_regression, y_pred_xgb_regressor, "XGBoost Regressor")

    # =================================================================
    # Print final summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (Test Set)")
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics).T
    print(summary_df.to_string())
    if poly_reg_metrics:
        print("\n  Polynomial Regression:")
        for k, v in poly_reg_metrics.items():
            print(f"    {k}: {v}")
    if lin_reg_metrics:
        print("\n  Linear Regression:")
        for k, v in lin_reg_metrics.items():
            print(f"    {k}: {v}")
    print("\n  XGBoost Regression:")
    for k, v in reg_metrics.items():
        print(f"    {k}: {v}")
    print("=" * 60)
