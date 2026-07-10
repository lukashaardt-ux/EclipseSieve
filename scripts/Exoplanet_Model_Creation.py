'''
Creates the four models including EclipseSieve.
This script trains and tests the models as well as providing an evaluation of the models.
'''

import pandas as pd
import numpy as np 
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score,confusion_matrix, classification_report, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model_utils import BalancedXGBClassifier
from paths import EXOPLANET_FEATURES_CSV, ECLIPSE_SIEVE_PKL, figure_path, results_path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')


#Fetching Data

df=pd.read_csv(EXOPLANET_FEATURES_CSV)

#Create New Columns in DF

df['snr_x_depth']=df['snr']*df['depth']
df['sec_over_depth']=df['sec_eclipse_depth']/(df['depth']+1e-9)
df['duration_over_period']=df['duration']/df['period']
df['odd_even_x_sec']=df['odd_even_diff']*df['sec_eclipse_depth']
df['transit_shape_metric'] = df['depth'] / (df['duration'] + 1e-9)
df['flux_variability'] = df['std_oot'] / (df['archive_snr'] + 1e-9)
df['sec_depth_significance'] = df['sec_eclipse_depth'] / (df['out_of_transit_rms'] + 1e-9)

# Feature suites (For Phase 2 and beyond)
'''
feature_domains = {
    'Stellar Context': [
        'planet_radius', 'rp_over_rs', 'stellar_logg', 'stellar_teff'
    ],
    'Transit Geometry': [
        'archive_duration', 'depth', 'depth_ratio', 'duration', 
        'flux_asymmetry', 'kurtosis', 'skewness', 'symmetry'
    ],
    'Noise and Signal': [
        'archive_snr', 'bls_power', 'cdpp', 'max_mes', 'max_ses', 
        'out_of_transit_rms', 'period', 'snr', 'std_oot', 'transit_snr_bls'
    ],
    'Eclipsing Binaries & Interference': [
        'duration_over_period', 'impact_parameter', 'odd_even_diff', 
        'odd_even_x_sec', 'sec_eclipse_depth', 'sec_over_depth', 'snr_x_depth'
    ]
}

current_suite='Eclipsing Binaries & Interference'
# Intiating Phase 2
features = feature_domains[current_suite]
'''

#Create Features
features=[f for f in df.columns if f not in ['label','kepid','disposition']] #Used for Phase 1

X=df[features]
y=df['label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#Create Pipeline



RF_pipeline=Pipeline([
    ('model',RandomForestClassifier(class_weight='balanced',random_state=42))
    ])

xgb_pipeline = Pipeline([
    ('model', BalancedXGBClassifier(
        random_state=42,
        eval_metric='logloss'
    ))
])

LR_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', random_state=42, max_iter=5000))
])

SVM_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(probability=True, class_weight='balanced', random_state=42, max_iter=5000))
])

SVM_pipeline_eval = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(probability=False, class_weight='balanced', random_state=42, max_iter=5000))
])

#Tuning Models

# Hyperparameter grids for tuning
RF_param_grid = {
    'model__n_estimators': [200, 500],
    'model__max_depth': [None, 20],
    'model__min_samples_leaf': [1, 4],
    'model__max_features': ['sqrt']
}

XGB_param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

LR_param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__class_weight': ['balanced', None],
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [5000]
}

SVM_param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': ['scale', 'auto', 0.01, 0.1],
    'model__kernel': ['linear', 'rbf']
}

def tune_model_RF(X_train,y_train):
    param_grid=RF_param_grid
    grid_search=GridSearchCV(RF_pipeline,
                            param_grid,
                            cv=StratifiedKFold(5),
                            scoring='f1',
                            n_jobs=-1
    )                    
    grid_search.fit(X_train,y_train)

    print(f'Optimal Random Forest Parameters: {grid_search.best_params_}')


    return grid_search.best_estimator_

def tune_model_XGB(X_train, y_train):
    param_grid = XGB_param_grid
    grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f'Optimal XGBoost Parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_

def tune_model_LR(X_train, y_train):
    param_grid = LR_param_grid

    grid_search = GridSearchCV(
        estimator=LR_pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f'Optimal Logistic Regression Parameters: {grid_search.best_params_}')

    return grid_search.best_estimator_

def tune_model_SVM(X_train, y_train):
    param_grid = SVM_param_grid

    grid_search = GridSearchCV(
        estimator=SVM_pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(5),
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f'Optimal SVM Parameters: {grid_search.best_params_}')

    return grid_search.best_estimator_

RF=tune_model_RF(X_train,y_train)
xgb = tune_model_XGB(X_train, y_train)
LR = tune_model_LR(X_train, y_train)
SVM = tune_model_SVM(X_train, y_train)


models = {
    'Random Forest': RF,
    'XGBoost': xgb,
    'Logistic Regression': LR,
    'SVM': SVM
}

print('\n---5-FOLD CROSS-VALIDATION RESULTS ---')

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_searches = {
    'Random Forest': GridSearchCV(RF_pipeline, RF_param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1),
    'XGBoost': GridSearchCV(xgb_pipeline, XGB_param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1),
    'Logistic Regression': GridSearchCV(LR_pipeline, LR_param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1),
    'SVM': GridSearchCV(SVM_pipeline_eval, SVM_param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
}

nested_cv_results = {}

for name, search_estimator in nested_searches.items():
    print(f'Running nested CV for {name}...')
    scores = cross_validate(search_estimator,
                            X,
                            y,
                            cv=outer_cv,
                            scoring=['accuracy', 'f1', 'roc_auc'],
                            n_jobs=1
                            )
    nested_cv_results[name] = scores
    print(f'\n{name} — Nested CV:')
    print(f'  Mean Accuracy: {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}')
    print(f'  Mean F1 Score: {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}')
    print(f'  Mean ROC AUC: {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}')


#Important Features
RF_importances=RF.named_steps['model'].feature_importances_
XGB_importances=xgb.named_steps['model'].feature_importances_
LR_importances = np.abs(LR.named_steps['model'].coef_[0])
feature_names=X.columns

RF_importance_df=pd.DataFrame({
    'feature': feature_names,
    'importance': RF_importances
    }).sort_values(by='importance', ascending=False)

XGB_importance_df=pd.DataFrame({
    'feature': feature_names,
    'importance': XGB_importances
    }).sort_values(by='importance', ascending=False)

LR_importance_df=pd.DataFrame({
    'feature': feature_names,
    'importance': LR_importances
    }).sort_values(by='importance', ascending=False)


#Evaluate Model

def evaluate_model(best_model,X_test,y_test):
    probs = best_model.predict_proba(X_test)[:,1]
    predictions= (probs > 0.5).astype(int)
    cm=confusion_matrix(y_test,predictions)
    accuracy=accuracy_score(y_test,predictions)
    f1=f1_score(y_test,predictions)
    roc_auc=roc_auc_score(y_test,probs)

    print(classification_report(y_test,predictions))

    return f1,roc_auc,accuracy,cm,probs

f1_RF,roc_auc_RF,accuracy_RF,cm_RF,probs_RF=evaluate_model(RF,X_test,y_test)
f1_XGB,roc_auc_XGB,accuracy_XGB,cm_XGB,probs_XGB=evaluate_model(xgb,X_test,y_test)
f1_LR,roc_auc_LR,accuracy_LR,cm_LR,probs_LR=evaluate_model(LR,X_test,y_test)
f1_SVM,roc_auc_SVM,accuracy_SVM,cm_SVM,probs_SVM=evaluate_model(SVM,X_test,y_test)

#RF Stats
print(f'The accuracy of the RF is {round(accuracy_RF*100,2)}%')
print(f'F1 score: {f1_RF:.3f}')
print(f'ROC-AUC: {roc_auc_RF:.3f}')

#XGB Stats
print(f'The accuracy of the XGB is {round(accuracy_XGB*100,2)}%')
print(f'F1 score: {f1_XGB:.3f}')
print(f'ROC-AUC: {roc_auc_XGB:.3f}')

#LR Stats
print(f'The accuracy of the LR is {round(accuracy_LR*100,2)}%')
print(f'F1 score: {f1_LR:.3f}')
print(f'ROC-AUC: {roc_auc_LR:.3f}')

#SVM Stats
print(f'The accuracy of the SVM is {round(accuracy_SVM*100,2)}%')
print(f'F1 score: {f1_SVM:.3f}')
print(f'ROC-AUC: {roc_auc_SVM:.3f}')

X_test_reset = X_test.reset_index(drop=False)
orig_idx = X_test.index


print('RF Beliefs:')
for i in range(0,min(20,len(probs_RF))):
    idx = orig_idx[i]
    prob=probs_RF[i]
    print(f'KIC {int(df.loc[idx,'kepid'])}')
    print(f' Chance of being an exoplanet: {prob*100:.1f}%')
    if df.loc[idx,'label'] == 1:
        print('Actual: Exoplanet')
    else:
        print('Actual: Not Exoplanet')

print('XGB Beliefs:')
for i in range(0,min(20,len(probs_XGB))):
    idx = orig_idx[i]
    prob=probs_XGB[i]
    print(f'KIC {int(df.loc[idx,'kepid'])}')
    print(f' Chance of being an exoplanet: {prob*100:.1f}%')
    if df.loc[idx,'label'] == 1:
        print('Actual: Exoplanet')
    else:
        print('Actual: Not Exoplanet')
'''
print('LR Beliefs:')
for i in range(0,min(20,len(probs_LR))):
    idx = orig_idx[i]
    prob=probs_LR[i]
    print(f"KIC {int(df.loc[idx,'kepid'])}")
    print(f" Chance of being an exoplanet: {prob*100:.1f}%")
    if df.loc[idx,'label'] == 1:
        print('Actual: Exoplanet')
    else:
        print('Actual: Not Exoplanet')

print('SVM Beliefs:')
for i in range(0,min(20,len(probs_SVM))):
    idx = orig_idx[i]
    prob=probs_SVM[i]
    print(f"KIC {int(df.loc[idx,'kepid'])}")
    print(f" Chance of being an exoplanet: {prob*100:.1f}%")
    if df.loc[idx,'label'] == 1:
        print('Actual: Exoplanet')
    else:
        print('Actual: Not Exoplanet')
'''
#Visualization

#Confusion Matrices
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm,annot=True,fmt='d',xticklabels=['False Positive','Confirmed'],yticklabels=['False Positive','Confirmed'],cmap='coolwarm')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(figure_path(f'{model_name.replace(' ', '_')}_Confusion_Matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(cm_RF, 'Random Forest')
plot_confusion_matrix(cm_XGB, 'XGBoost')
plot_confusion_matrix(cm_LR, 'Logistic Regression')
plot_confusion_matrix(cm_SVM, 'SVM')

#Importances
def plot_importances(importance_df, model_name):
    plt.figure(figsize=(10,7))
    plt.barh(importance_df['feature'],importance_df['importance'] * 100,edgecolor='black',color='purple')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Importance of Features for {model_name}')
    plt.savefig(figure_path(f'{model_name.replace(' ', '_')}_Importance.png'), dpi=300, bbox_inches='tight')
    plt.show()


plot_importances(RF_importance_df, 'Random Forest')
plot_importances(XGB_importance_df, 'XGBoost')
plot_importances(LR_importance_df, 'Logistic Regression')

try:
    svm_importances = np.abs(SVM.named_steps['model'].coef_[0])
    SVM_importance_df=pd.DataFrame({
        'feature': feature_names,
        'importance': svm_importances
    }).sort_values(by='importance', ascending=False)
    
    plot_importances(SVM_importance_df, 'SVM')

except AttributeError:
    print('\n[NOTE] SVM selected a non-linear kernel.')
    print('Non-linear SVMs do not have linear coefficients to plot.')
    print('We will evaluate its feature importance using Permutation Feature Importance in the next phase!\n')

#ROC curve
roc_models = [
    ('Random Forest', probs_RF, 'purple'),
    ('XGBoost', probs_XGB, 'royalblue'),
    ('Logistic Regression', probs_LR, 'orange'),
    ('SVM', probs_SVM, 'red'),
]

fig, ax = plt.subplots(figsize=(8, 6))
for label, probs, color in roc_models:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, label=f'{label} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
ax.set_title('ROC Curve — EclipseSieve Model Comparison')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.savefig(figure_path('roc_curve.png'), dpi=300, bbox_inches='tight')
plt.show()

#Transfer model to candidate tester file

joblib.dump({
    'RF':       RF,
    'XGB':      xgb,
    'LR':       LR,
    'SVM':      SVM,
    'features': features
}, ECLIPSE_SIEVE_PKL)

def log_experiment_run(model_records, feature_list, file_path=results_path('pipeline_experiment_log.csv')):

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    print('\n--- EXPERIMENT LOGGING SYSTEM ---')
    run_notes = ''  # You can add any specific notes about this run here (e.g., "Added new features", "Tuned SVM with RBF kernel", etc.)
    
    rows = []
    for model_name, metrics in model_records.items():
        rows.append({
            'Timestamp': timestamp,
            'Model_Architecture': model_name,
            'Accuracy': round(metrics['accuracy'], 4),
            'F1_Score': round(metrics['f1'], 4),
            'ROC_AUC': round(metrics['auc'], 4),
            'Total_Features': len(feature_list),
            'Feature_Suite': ', '.join(feature_list[:5]) + (f' ... (+{len(feature_list)-5} more)' if len(feature_list) > 5 else ''),
            'Engineering_Notes': run_notes
        })
    
    new_data = pd.DataFrame(rows)
    file_exists = os.path.exists(file_path)
    
    new_data.to_csv(file_path, mode='a', header=not file_exists, index=False)
    print(f"[SUCCESS] Metrics successfully recorded inside '{file_path}'!\n")

current_run_metrics = {
    'Random Forest': {'accuracy': accuracy_RF, 'f1': f1_RF, 'auc': roc_auc_RF},
    'XGBoost':       {'accuracy': accuracy_XGB, 'f1': f1_XGB, 'auc': roc_auc_XGB},
    'Logistic Regression': {'accuracy': accuracy_LR, 'f1': f1_LR, 'auc': roc_auc_LR},
    'SVM':             {'accuracy': accuracy_SVM, 'f1': f1_SVM, 'auc': roc_auc_SVM}
}

log_experiment_run(current_run_metrics, features)