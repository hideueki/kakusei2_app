# ライブラリのインポート
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, log_loss

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# データセットの取得
df = pd.read_csv("/content/drive/MyDrive/RARPリンパ節転移_機械学習/拡大郭清+転移あり.csv", encoding='utf-8')
columns = [ 'iPSA(ng/ml)',  'cT3≦', 'core%', 'ISUP_grade', "リンパ節転移あり"]
df = df[columns]

# エンコード
oe = OrdinalEncoder()
# oe.set_output(transform="pandas")
a = np.array(df["ISUP_grade"]).reshape(-1, 1)
oe.fit(a)
df['ISUP_grade'] = oe.transform(a)

# データセットの準備
X = df.drop(["リンパ節転移あり"], axis=1)
y = df['リンパ節転移あり']

# 欠損値を最頻値または中央値で埋める
for column in X.columns:
    # 最頻値を計算
    mode_values = X[column].mode()

    # もし最頻値が1つしかなければそれを使用、そうでなければ中央値を使用
    if mode_values.shape[0] == 1:
        X[column].fillna(mode_values[0], inplace=True)
    else:
        X[column].fillna(X[column].median(), inplace=True)
        

# 最適なハイパーパラメータを取得
best_params = {'max_depth': 3, 'num_leaves': 2, 'reg_alpha': 0.03}

# 最適なハイパーパラメータを使用してLightGBMの新しいインスタンスを作成
lgbmc_best = lgb.LGBMClassifier(**{k.split("__")[1]: v for k, v in best_params.items() if "model__" in k})

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# モデルのトレーニング
lgbmc_best.fit(X_train, y_train)
y_proba_lgbmc = lgbmc_best.predict_proba(X_test)
y_pred_lgbmc = lgbmc_best.predict(X_test)

# 予測精度を確認
cm = confusion_matrix(y_test, y_pred_lgbmc)

# 学習済みモデルを保存
joblib.dump(lgbmc_best, 'src/lgbmc.pkl', compress=True)