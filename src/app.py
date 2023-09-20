# 必要なモジュールのインポート
import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np

#　学習済みモデルをもとに推論する関数
def predict(x):
    # 学習済みモデル（kakusei.pkl）を読み込み
    model = joblib.load('/Users/hideto/Desktop/kakusei_app/src/kakusei.pkl')
    x = x.reshape(1,-1)
    pred_label = model.predict(x)
    return pred_label

def getName(label):
    if label == 0:
        return '転移がある可能性は低いです'
    elif label == 1:
        return '転移がある可能性が高いです'
    else:
        return 'Error'

# Flaskのインスタンスを作成
app = Flask(__name__)

# 入力フォームの設定
class KakuseiForm(Form):
    
    PSA = FloatField('iPSAの値',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=100, message='数値を入力してください')])

    T3  = FloatField('cT3かどうか（0 or 1）',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=1, message='0か1の数値を入力してください')])

    CorePercent = FloatField('陽性コアの割合(%)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=99, message='数値を入力してください')])

    IsupGrade  = FloatField('ISUP_grade(1~5)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=1, max=5, message='1〜5の数値を入力してください')])

    # html 側で表示する submit ボタンの設定
    submit = SubmitField('判定')

# URL にアクセスがあった場合の挙動の設定# ... [以前のコードは変更なし]

@app.route('/', methods=['GET', 'POST'])
def predicts():
    # TForms で構築したフォームをインスタンス化
    kakusei_form_instance = KakuseiForm(request.form)  # この行を変更
        
    # POST メソッドの定義
    if request.method == 'POST':

    # 条件に当てはまらない場合
        if not kakusei_form_instance.validate():
            return render_template('index.html', forms=kakusei_form_instance)

    # 条件に当てはまる場合の、推論を実行
        else:
            PSA = float(request.form['PSA'])
            T3 = int(request.form['T3'])  # 修正されたキー名
            CorePercent = float(request.form['CorePercent'])  # 修正されたキー名
            IsupGrade = int(request.form['IsupGrade'])  # 修正されたキー名

            # 入力された値を np.array に変換して推論
            x = np.array([[PSA, T3, CorePercent, IsupGrade]])
            pred = predict(x)
            result_Name_ = getName(pred)
            return render_template('result.html', result_Name=result_Name_)


    # GET 　メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=kakusei_form_instance)  # この行を変更
    
# アプリケーションの実行
if __name__ == '__main__':
    app.debug = True
    app.run()