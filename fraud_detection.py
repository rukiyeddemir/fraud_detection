import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
!pip install optuna
import optuna

### Ayarlar
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', None)     # Tüm satırları göster

### Veri setini yükleme
train_transaction = pd.read_csv(r"/content/train_transaction.csv")
train_identity = pd.read_csv(r"/content/train_identity.csv")
train_left = train_transaction.merge(train_identity, on='TransactionID', how='left')

### Eksik veri oranları
((train_left.isnull().sum() / len(train_left)) * 100)  #.sort_values(ascending=False)

### Eksik değer silme
threshold = 40
missing_percentage = (train_left.isnull().sum() / len(train_left)) * 100
columns_to_drop = missing_percentage[(missing_percentage > threshold) & (missing_percentage.index != 'DeviceType')].index
train_left = train_left.drop(columns=columns_to_drop)
train_left = train_left.drop(columns=['P_emaildomain'])

### Değişken türlerini belirleme
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünen kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: DataFrame
            Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
            Numerik fakat kategorik olan değişkenler için sınır eşik değeri
        car_th: int, optional
            Kategorik fakat kardinal değişkenler için sınır eşik değeri

    Returns
    ------
        cat_cols: list
            Kategorik değişken listesi
        num_cols: list
            Numerik değişken listesi
        cat_but_car: list
            Kategorik görünümlü kardinal değişken listesi
    """
    # Kategorik değişkenler
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerik değişkenler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

grab_col_names(train_left)

cat_cols, num_cols, cat_but_car = grab_col_names(train_left)

### Değişkenlerin unique değerleri
for col in cat_cols:
    print(f"{col} değişkeninin benzersiz değerleri: {train_left[col].unique()}")

for col in cat_but_car:
    print(f"{col} değişkeninin benzersiz değerleri: {train_left[col].unique()}")

for col in num_cols:
    print(f"{col} değişkeninin benzersiz değerleri: {train_left[col].unique()}")

### Eksik değer doldurma ve One Hot Encoding
train_left[['card4', 'card6', 'M6', 'DeviceType']] = train_left[['card4', 'card6', 'M6', 'DeviceType']].fillna('unknown')
train_left = pd.get_dummies(train_left, columns=['ProductCD', 'card4', 'card6', 'DeviceType', 'M6'])

numeric_columns = train_left.select_dtypes(include=[np.number]).columns
train_left[numeric_columns] = train_left[numeric_columns].apply(lambda x: x.fillna(x.median()))

scaler = StandardScaler()
train_left[num_cols] = scaler.fit_transform(train_left[num_cols])

### Test setini oluşturma
test_identity = pd.read_csv(r"/content/test_identity.csv")
test_transaction = pd.read_csv(r"/content/test_transaction.csv")
test_left = test_transaction.merge(test_identity, on='TransactionID', how='left')

train_columns = set(train_left.columns)
test_columns = set(test_left.columns)

# Test veri setinde eğitim veri setinde bulunmayan kolonları belirleme
extra_columns = test_columns - train_columns

# Test veri setinde bu ekstra kolonları silme
test_left = test_left.drop(columns=extra_columns)

### Model oluşturma
y = train_left["isFraud"]
X = train_left.drop(["TransactionID", "isFraud"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

### XGBoost ile modelleme
def fit_xgbm(X_train, y_train, X_test, y_test, n_trials=12):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
        }
        model = XGBClassifier(**params)
        model.set_params(early_stopping_rounds=200)
        accuracies = []
        for train_index, val_index in kfold.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
            model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=True)
            preds = model.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, preds)
            accuracies.append(accuracy)
        return np.mean(accuracies)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print("Best Parameters for XGBoost: ", best_params)

    # En iyi parametreler ile modeli başlat
    best_model = XGBClassifier(**best_params)
    best_model.set_params(early_stopping_rounds=200)

    def evaluate_cross_val(X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        accuracies, precisions, recalls, f2_scores = [], [], [], []
        for train_index, val_index in kfold.split(X):
            X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
            y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
            best_model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=True)
            preds = best_model.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, preds)
            precision = precision_score(y_val_cv, preds)
            recall = recall_score(y_val_cv, preds)
            f2 = fbeta_score(y_val_cv, preds, beta=2)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f2_scores.append(f2)
        return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f2_scores)

    train_accuracy, train_precision, train_recall, train_f2 = evaluate_cross_val(X_train, y_train)
    test_accuracy, test_precision, test_recall, test_f2 = evaluate_cross_val(X_test, y_test)

    # Final modeli early_stopping_rounds ile eğit
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    test_predictions = best_model.predict(X_test)

    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f2': train_f2,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f2': test_f2
    }

    return best_model, metrics, test_predictions

best_model, metrics, test_predictions = fit_xgbm(X_train, y_train, X_test, y_test)

print("Model Metrics: ", metrics)
