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
    y_test_cls = test_df["Direction"].values
    y_test_reg = test_df["Next_Close"].values
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
            y_test_cls, y_pred_lr, "Logistic Regression")

    # =================================================================
    # Model 2 — XGBoost Classifier + Regressor
    # =================================================================
    print("=" * 60)
    print("[evaluate] Model 2 — XGBoost")
    print("=" * 60)

    xgb_clf = _load_artifact("xgboost_classifier.pkl", save_dir)
    xgb_reg = _load_artifact("xgboost_regressor.pkl", save_dir)
    xgb_scaler = _load_artifact("xgboost_scaler.pkl", save_dir)

    X_test_xgb = xgb_scaler.transform(X_test)

    # Classification
    y_pred_xgb_cls = xgb_clf.predict(X_test_xgb)
    all_metrics["XGBoost Classifier"] = _classification_metrics(
        y_test_cls, y_pred_xgb_cls, "XGBoost Classifier")

    # Regression — model predicts pct change; convert back to prices
    y_pred_pct = xgb_reg.predict(X_test_xgb)
    y_pred_xgb_reg = test_close * (1 + y_pred_pct / 100)
    reg_metrics = _regression_metrics(
        y_test_reg, y_pred_xgb_reg, "XGBoost Regressor")

    # =================================================================
    # Print final summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (Test Set)")
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics).T
    print(summary_df.to_string())
    print("\n  XGBoost Regression:")
    for k, v in reg_metrics.items():
        print(f"    {k}: {v}")
    print("=" * 60)
