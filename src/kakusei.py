# ライブラリのインポート
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import pandas as pd

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
df = pd.read_csv("/Users/hideto/Desktop/kakusei_app/src/拡大郭清+転移あり.csv", encoding='utf-8')
columns = ['Number of positive core', 'Negative core', 'iPSA(ng/ml)', 'cT3≦', 'GS2', 'ISUP_grade', "リンパ節転移あり"]
df = df[columns]

# データセットの準備
X = df.drop(["リンパ節転移あり"], axis=1)
y = df['リンパ節転移あり']

# モデルのインスタンス作成
logi = LogisticRegression(random_state=0, C=0.1, class_weight=None, penalty='l2')

# パイプラインの定義
pipeline = Pipeline(steps=[("preprocessing", StandardScaler()), ("model", logi)])

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# モデルのトレーニング
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

# モデルを使用してテストデータの陽性確率を予測
proba = pipeline.predict_proba(X_test)

# カットオフ値を0.2に設定して予測ラベルを生成
# proba[:, 1] は陽性クラス（'リンパ節転移あり'）に対する確率
pred_cutoff_0_2 = (proba[:, 1] > 0.2).astype(int)

# 予測精度を確認（カットオフ値0.2を使用）
print(classification_report(y_test, pred_cutoff_0_2))

# 学習済みモデルを保存
joblib.dump(pipeline, 'src/kakusei.pkl', compress=True)
