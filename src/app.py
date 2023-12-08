# 必要なモジュールのインポート
import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np

#　学習済みモデルをもとに推論する関数
def predict(x):
    # 学習済みモデル（kakusei.pkl）を読み込み
    model = joblib.load('src/kakusei.pkl')
    # model = joblib.load('./kakusei.pkl')
    x = x.reshape(1,-1)
    # pred_label = model.predict(x)
    proba = model.predict_proba(x)
    rounded_proba = np.around(proba[:, 1], decimals=4)
    # pred_cutoff_0_2 = (proba[:, 1] > 0.2).astype(int)
    return rounded_proba

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
    
    NumberofPositiveCore = FloatField('Number of positive cores',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=100, message='Please enter a number')])

    NumberofNegativeCore  = FloatField('Number of negative cores',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=100, message='Please enter a number')])
    
    PSA  = FloatField('PSA',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=100, message='Please enter a number')])
    
    cT3  = FloatField('cT3 or cT4: No:0, Yes:1',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=1, message=': Please enter 0 or 1')])
    

    SecondaryGS = FloatField('Secondary Gleason Score',
                    [validators.InputRequired(),
                    validators.NumberRange(min=0, max=5, message='Please enter a number of 0~5')])

    IsupGrade  = FloatField('ISUP grade(1~5)',
                    [validators.InputRequired(),
                    validators.NumberRange(min=1, max=5, message='Please enter a number of 1~5')])

    # html 側で表示する submit ボタンの設定
    submit = SubmitField('Enter')

# URL にアクセスがあった場合の挙動の設定# ... [以前のコードは変更なし]

@app.route('/', methods=['GET', 'POST'])
def predicts():
    # TForms で構築したフォームをインスタンス化
    kakusei_form_instance = KakuseiForm(request.form)
        
    # POST メソッドの定義
    if request.method == 'POST':

    # 条件に当てはまらない場合
        if not kakusei_form_instance.validate():
            return render_template('index.html', forms=kakusei_form_instance)

    # 条件に当てはまる場合の、推論を実行
        else:
            NumberofPositiveCore = int(request.form['NumberofPositiveCore']) 
            NumberofNegativeCore = int(request.form['NumberofNegativeCore']) 
            PSA = float(request.form['PSA'])
            cT3 = int(request.form['cT3'])  
            SecondaryGS = int(request.form['SecondaryGS'])  
            IsupGrade = int(request.form['IsupGrade']) 

            # 入力された値を np.array に変換して推論
            x = np.array([[NumberofPositiveCore, NumberofNegativeCore, PSA, cT3, SecondaryGS, IsupGrade]])
            pred = predict(x)
            # result_Name_ = getName(pred)
            return render_template('result.html', result_Name=pred)


    # GET 　メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=kakusei_form_instance)  # この行を変更
    
# アプリケーションの実行
if __name__ == '__main__':
    app.debug = True
    app.run()