# --- Import necessary libraries ---
import random
import dalex as dx                           # For model explainability (SHAP, etc.)
from dbnomics import fetch_series            # To fetch IMF/WB economic datasets
import pandas as pd
from scikeras.wrappers import KerasRegressor # Keras model wrapper for sklearn
from sklearn.preprocessing import StandardScaler
import lime.lime_tabular                     # For LIME tabular explanations
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.utils import validation

# ===============================================================
# === 1. DATA RETRIEVAL FROM INTERNATIONAL SOURCES (IMF, WB) ===
# ===============================================================

# IMF and World Bank country codes
imf_codes = [...]  # IMF codes (list of country identifiers)
wb_codes = [...]   # WB codes (3-letter ISO country identifiers)

# --- Select IMF economic indicators for inflation analysis ---
# Example indicators include CPI, labor, banking, external balance, etc.
indicators_1 = [
    'PCPI_PC_CP_A_PT',  # Consumer Price Index (CPI)
    'LP_PE_NUM',        # Employment
    'BCAXF_BP6_USD', 'RAXGFX_USD', 'BFDA_BP6_USD', 'BK_DB_BP6_USD',
    'RAFA_G_USD', 'LUR_PT'  # Labor unemployment rate
]

# --- Fetch IMF International Financial Statistics (IFS) dataset ---
ifs_data_1 = fetch_series(
    "IMF", "IFS",
    max_nb_series=3000,
    dimensions={"REF_AREA": imf_codes, "FREQ": ["A"], "INDICATOR": indicators_1}
)

# --- Pivot the fetched IMF dataset to a wide format ---
cpi_analysis_df_1 = ifs_data_1.pivot(
    index=["Reference Area", "period", "original_period"],
    columns="INDICATOR",
    values="value"
)
cpi_analysis_df_1.reset_index(inplace=True)

# Drop rows with missing values
print(cpi_analysis_df_1.isnull().sum())
cpi_analysis_df_1.dropna(axis=0, how='any', inplace=True)

# --- Compute per capita variables for USD-based indicators ---
for cc in indicators_1[2:-1]:
    cpi_analysis_df_1[cc.replace("USD", "USD_pc")] = cpi_analysis_df_1[cc] / cpi_analysis_df_1['LP_PE_NUM']

# Replace old indicator names with per capita versions
indicators_1 = [ind.replace("USD", "USD_pc") for ind in indicators_1]

# ===========================================================
# === 2. WORLD BANK WDI AND WGI DATA FOR ADDITIONAL INPUTS ===
# ===========================================================

# --- World Development Indicators (economic) ---
wdi_data_1 = pd.DataFrame()
indicators_2 = ['AG.LND.ARBL.ZS', 'BN.CAB.XOKA.GD.ZS', 'BX.GSR.CMCP.ZS', 'NE.CON.GOVT.KD.ZG', 'NE.IMP.GNFS.ZS']
series = [f'WB/WDI/A-{ind}-{wb}' for wb in wb_codes for ind in indicators_2]

# Fetch data in manageable batches
batch_size = 50
for i in range(0, len(series)//batch_size + 1):
    print(i)
    x = fetch_series(series[i*batch_size:(i+1)*batch_size])
    wdi_data_1 = pd.concat([wdi_data_1, x], axis=0)

wdi_data_1.columns.values[wdi_data_1.columns == 'country (label)'] = "Reference Area"
wdi_data_1.reset_index(inplace=True)
cpi_analysis_df_2 = wdi_data_1.pivot(
    index=["Reference Area", "period", "original_period"],
    columns="indicator",
    values="value"
)
cpi_analysis_df_2.reset_index(inplace=True)
cpi_analysis_df_2.dropna(axis=0, how='any', inplace=True)

# --- World Governance Indicators (institutional quality) ---
wdi_data_2 = pd.DataFrame()
indicators_3 = ['CC.PER.RNK', 'GE.PER.RNK', 'PV.PER.RNK', 'RQ.PER.RNK']  # Control of corruption, governance, etc.
series = [f'WB/WGI/A-{ind}-{wb}' for wb in wb_codes for ind in indicators_3]

# Fetch WGI data in batches
batch_size = 50
for i in range(0, len(series)//batch_size + 1):
    print(i)
    x = fetch_series(series[i*batch_size:(i+1)*batch_size])
    wdi_data_2 = pd.concat([wdi_data_2, x], axis=0)

wdi_data_2.columns.values[wdi_data_2.columns == 'country (label)'] = "Reference Area"
wdi_data_2.reset_index(inplace=True)
cpi_analysis_df_3 = wdi_data_2.pivot(
    index=["Reference Area", "period", "original_period"],
    columns="indicator",
    values="value"
)
cpi_analysis_df_3.reset_index(inplace=True)
cpi_analysis_df_3.dropna(axis=0, how='any', inplace=True)

# ===================================================
# === 3. MERGE AND PREPARE COMPLETE ANALYTICAL SET ===
# ===================================================

# Drop redundant time columns
for df in [cpi_analysis_df_1, cpi_analysis_df_2, cpi_analysis_df_3]:
    df.drop(columns=['period'], inplace=True)

# Merge all IMF, WDI, and WGI datasets
cpi_analysis_full_df = pd.merge(
    cpi_analysis_df_1, cpi_analysis_df_2,
    on=['Reference Area', 'original_period'], how='inner'
)
cpi_analysis_full_df = pd.merge(
    cpi_analysis_full_df, cpi_analysis_df_3,
    on=['Reference Area', 'original_period'], how='inner'
).dropna()

# Create a combined index (country + year)
cpi_analysis_full_df['ind'] = cpi_analysis_full_df["Reference Area"] + "_" + cpi_analysis_full_df["original_period"]
cpi_analysis_full_df.set_index("ind", inplace=True)

# --- Define predictors (X) and target (y) ---
X = cpi_analysis_full_df[indicators_1[2:] + indicators_2 + indicators_3]
y = cpi_analysis_full_df[indicators_1[0]]  # CPI (inflation) target variable

# ====================================================
# === 4. BUILD AND TRAIN A NEURAL NETWORK REGRESSOR ===
# ====================================================

random.seed(25032005)
# Using full data for demonstration; in practice, train_test_split is preferred
X_train, X_test, y_train, y_test = X, X, y, y

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Keras model architecture ---
def build_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Regression output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Wrap model in scikit-learn interface ---
cpi_model = KerasRegressor(model=build_model, epochs=300, batch_size=15, verbose=True)
cpi_model.fit(X_train_scaled, y_train)

# --- Plot learning curve ---
plt.figure(figsize=(10, 6))
plt.plot(cpi_model.history_["loss"], label="Training Loss")
if "val_loss" in cpi_model.history_:
    plt.plot(cpi_model.history_["val_loss"], label="Validation Loss")
plt.title("CPI Neural Network Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("cpi_learning_curve.jpg")
plt.show()

# ======================================================
# === 5. FEATURE IMPORTANCE (Permutation Importance) ===
# ======================================================

perm_importance = permutation_importance(
    estimator=cpi_model,
    X=X.values,
    y=y,
    scoring="neg_mean_squared_error",
    n_repeats=10,
    random_state=25032005
)
print(perm_importance.importances_mean)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# --- Plot permutation feature importance ---
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Permutation Importance (? MSE)")
plt.title("Permutation Feature Importance - CPI Model")
plt.gca().invert_yaxis()
plt.savefig("cpi_vip_results.jpg")
plt.show()

# ==================================================
# === 6. LOCAL EXPLANATION WITH LIME (Poland 2020) ===
# ==================================================

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Inflation'],
    mode='regression'
)

# Select instance for Poland in 2020
poland_2020_index = X.index.get_loc('Poland_2020')
prediction = cpi_model.predict(X_test_scaled)[poland_2020_index]

# --- Generate LIME explanation for this case ---
explanation = explainer.explain_instance(
    X_test_scaled[poland_2020_index],
    cpi_model.predict,
    num_features=15
)

# Display LIME weights
exp = explanation.as_list()
feature_names, weights = zip(*exp)
print(exp)

# --- Plot LIME feature contributions ---
fig = plt.figure(figsize=(18, 6))
plt.barh(feature_names, weights, color=['blue', "red", "green"])
plt.xlabel('Feature contribution to predicted CPI change')
plt.title('LIME Explanation for CPI Prediction (Poland, 2020)')
plt.gca().invert_yaxis()
plt.show()
fig.savefig("cpi_lime_poland_2020.jpeg")

# ====================================================
# === 7. GLOBAL EXPLANATION WITH SHAP (via DALEX) ===
# ====================================================

dbnomic_fnn_exp = dx.Explainer(cpi_model, X, y, label="CPI SHAP Pipeline")

bd_poland_2020 = dbnomic_fnn_exp.predict_parts(X_test_scaled[poland_2020_index], type='shap')
pd.set_option('display.max_rows', 100)
print(bd_poland_2020.result.query('B==0'))

# --- Plot SHAP explanation for Poland 2020 ---
bd_poland_2020.plot()
plt.savefig("cpi_shap_poland_2020.jpeg")

# =====================================================
# === 8. PARTIAL DEPENDENCE PLOTS (Feature Effects) ===
# =====================================================

for i in range(3):
    PartialDependenceDisplay.from_estimator(
        estimator=cpi_model,
        X=X.values[np.random.permutation(range(X_test_scaled.shape[0]))[:100],],
        features=list(range(X.shape[1]))[5 * i:5 * (i + 1)],
        feature_names=X.columns,
        kind="both",  # average + individual (ICE) curves
        random_state=25032005
    )
    plt.savefig(f"cpi_pdp_results_{i}.jpg")
    plt.show()
