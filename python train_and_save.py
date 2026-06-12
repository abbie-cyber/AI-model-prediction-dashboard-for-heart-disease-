{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "468ebba2-fbb6-4399-bf3a-ae6a2864d7bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Random Forest...\n",
      "Training XGBoost...\n",
      "✅ Models and data successfully saved!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import warnings\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OrdinalEncoder, StandardScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import xgboost as xgb\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "\n",
    "# 1. Load Data\n",
    "df = pd.read_csv(\"Heart.csv\", na_values=[\"NA\", \"?\", \"\"])\n",
    "df[\"AHD\"] = df[\"AHD\"].map({\"No\": 0, \"Yes\": 1})\n",
    "x = df.drop(columns=[\"AHD\", \"HD\"])\n",
    "y = df[\"AHD\"]\n",
    "\n",
    "# 2. Preprocessing\n",
    "num_features = [\"Age\", \"RestBP\", \"Chol\", \"MaxHR\", \"Oldpeak\", \"Ca\"]\n",
    "cat_features = [\"Sex\", \"ChestPain\", \"Fbs\", \"RestECG\", \"ExAng\", \"Slope\", \"Thal\"]\n",
    "\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "    (\"encoder\", OrdinalEncoder(handle_unknown=\"use_encoded_value\", unknown_value=-1))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    (\"num\", numeric_transformer, num_features),\n",
    "    (\"cat\", categorical_transformer, cat_features)\n",
    "], remainder=\"drop\")\n",
    "\n",
    "# 3. Split Data\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=SEED, stratify=y)\n",
    "\n",
    "# 4. Build Pipelines\n",
    "pos_weight = (y_train == 0).sum() / (y_train == 1).sum()\n",
    "\n",
    "rf_pipeline = Pipeline(steps=[\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"rf\", RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight=\"balanced\"))\n",
    "])\n",
    "\n",
    "xgb_pipeline = Pipeline(steps=[\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"xgb\", xgb.XGBClassifier(random_state=SEED, eval_metric=\"logloss\", n_jobs=-1, scale_pos_weight=pos_weight))\n",
    "])\n",
    "\n",
    "# 5. Train Models\n",
    "print(\"Training Random Forest...\")\n",
    "rf_pipeline.fit(x_train, y_train)\n",
    "print(\"Training XGBoost...\")\n",
    "xgb_pipeline.fit(x_train, y_train)\n",
    "\n",
    "# 6. Save Artifacts for Streamlit\n",
    "joblib.dump(rf_pipeline, \"rf_model.pkl\")\n",
    "joblib.dump(xgb_pipeline, \"xgb_model.pkl\")\n",
    "joblib.dump(x_test, \"x_test.pkl\")\n",
    "joblib.dump(y_test, \"y_test.pkl\")\n",
    "joblib.dump(num_features, \"num_features.pkl\")\n",
    "joblib.dump(cat_features, \"cat_features.pkl\")\n",
    "\n",
    "print(\"✅ Models and data successfully saved!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5a16662-e463-4380-8749-f983950291b9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
