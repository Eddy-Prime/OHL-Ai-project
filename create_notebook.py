import json

nb_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div style=\"text-align:center;padding:40px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:10px;color:white;margin-bottom:30px;\">\n",
    "<h1 style=\"margin:0;font-size:44px;font-weight:bold;\">OHL Football Attendance Prediction</h1>\n",
    "<p style=\"margin:15px 0 0 0;font-size:18px;opacity:0.9;\">Complete End-to-End Modeling Project</p>\n",
    "</div>\n",
    "\n",
    "## Project Overview\n",
    "\n",
    "This notebook executes the complete attendance prediction pipeline on real data and displays all generated outputs.\n",
    "\n",
    "The pipeline trains and compares models: mean baseline, opponent mean baseline, linear regression, random forest, and XGBoost.\n",
    "\n",
    "All features use only information available before kickoff, ensuring strict temporal ordering and no leakage."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quick Start\n",
    "\n",
    "Run cells from top to bottom to execute the training pipeline, load results, and generate predictions on new matches."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import json, sys, pandas as pd\n",
    "from IPython.display import display, Markdown, Image, HTML\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.width', None)\n",
    "\n",
    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
    "if str(project_root) not in sys.path:\n",
    "    sys.path.append(str(project_root))\n",
    "\n",
    "from src.train import run_pipeline\n",
    "from src.predict import predict_from_file\n",
    "from src.config import OUTPUTS_DIR, PREDICTIONS_DIR, FEATURE_IMPORTANCE_DIR, REPORTS_DIR, MODELS_DIR, BEST_MODEL_METADATA_PATH\n",
    "\n",
    "data_dir = Path(r'C:\\Users\\ASUS\\Desktop\\International Project\\Data')\n",
    "comparison_path = OUTPUTS_DIR / 'model_comparison.csv'\n",
    "test_predictions_path = PREDICTIONS_DIR / 'test_predictions.csv'\n",
    "new_predictions_path = PREDICTIONS_DIR / 'new_match_predictions.csv'\n",
    "summary_report_path = REPORTS_DIR / 'summary_report.txt'\n",
    "\n",
    "feature_importance_csv_candidates = [\n",
    "    FEATURE_IMPORTANCE_DIR / 'best_model_feature_importance.csv',\n",
    "    FEATURE_IMPORTANCE_DIR / 'random_forest_feature_importance.csv'\n",
    "]\n",
    "feature_importance_plot_candidates = [FEATURE_IMPORTANCE_DIR / 'top_feature_importance.png']\n",
    "prediction_plot_candidates = [\n",
    "    PREDICTIONS_DIR / 'actual_vs_predicted_best_model.png',\n",
    "    PREDICTIONS_DIR / 'attendance_over_time_test.png',\n",
    "    PREDICTIONS_DIR / 'attendance_distribution.png',\n",
    "    PREDICTIONS_DIR / 'residual_distribution_best_model.png'\n",
    "]\n",
    "\n",
    "print('Setup complete. Ready to execute training pipeline.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Run Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Starting training pipeline...')\n",
    "result = run_pipeline(data_dir=data_dir, tune_rf=False, tune_xgb=False, use_log_target=False)\n",
    "print('Training completed successfully.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "metadata = json.loads(BEST_MODEL_METADATA_PATH.read_text(encoding='utf-8'))\n",
    "summary_df = pd.DataFrame([{\n",
    "    'Best Model': metadata.get('best_model_name'),\n",
    "    'Train Rows': metadata.get('rows_train'),\n",
    "    'Test Rows': metadata.get('rows_test'),\n",
    "    'Target': metadata.get('target_column'),\n",
    "    'Features': len(metadata.get('features_used', []))\n",
    "}]).T\n",
    "summary_df.columns = ['Value']\n",
    "display(summary_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_df = pd.read_csv(comparison_path)\n",
    "display(comparison_df.style.format({'mae': '{:.2f}', 'rmse': '{:.2f}', 'r2': '{:.4f}'}).background_gradient(subset=['mae'], cmap='RdYlGn_r'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_importance_csv_path = next((p for p in feature_importance_csv_candidates if p.exists()), None)\n",
    "if feature_importance_csv_path:\n",
    "    feature_importance_df = pd.read_csv(feature_importance_csv_path)\n",
    "    display(feature_importance_df.head(15).style.bar(subset=['importance'], color='#667eea'))\n",
    "\n",
    "feature_plot_path = next((p for p in feature_importance_plot_candidates if p.exists()), None)\n",
    "if feature_plot_path:\n",
    "    display(Image(filename=str(feature_plot_path)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Future Match Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "template_path = project_root / 'data_examples' / 'new_match_input_template.csv'\n",
    "print('Scoring new matches...')\n",
    "new_predictions_runtime, prediction_summary = predict_from_file(input_file=template_path, data_dir=data_dir)\n",
    "print('Predictions complete.')\n",
    "\n",
    "summary_display = pd.DataFrame([prediction_summary]).T\n",
    "summary_display.columns = ['Value']\n",
    "display(summary_display)\n",
    "\n",
    "new_predictions_df = pd.read_csv(new_predictions_path)\n",
    "display(new_predictions_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if summary_report_path.exists():\n",
    "    report_text = summary_report_path.read_text(encoding='utf-8')\n",
    "    display(HTML(f'<div style=\"background:#f5f5f5;border-left:4px solid #667eea;padding:15px;border-radius:5px;\"><pre>{report_text}</pre></div>'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "This notebook demonstrates a complete, end-to-end attendance prediction system using leakage-safe features and XGBoost as the primary model. The best model has been automatically selected and saved for production inference."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('notebooks/MACHINE_learning.ipynb', 'w') as f:
    json.dump(nb_content, f, indent=1)

print('Notebook created successfully!')

