from flask import Flask, render_template, url_for, request, redirect, session, flash

# python基本ライブラリ
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import japanize_matplotlib

# matplotlib　表示
import base64
from io import BytesIO

# ファイル作成
import shutil
import datetime
import cv2
import os

# pytorch
import torch
from PIL import Image
from function import generation

app = Flask(__name__)
app.config["SECRET_KEY"] = "jfaogiehi2iw8jLD0ejJ"

# home画面
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# 1 偏差値
@app.route('/deviation_value', methods=['GET'])
def deviation_value_get():
    return render_template('1_deviation_value.html')

@app.route("/deviation_value", methods=['POST'])
def deviation_value_post():
    
    # 入力受け取り
    mu = request.form['mu']
    sd = request.form['sd']
    sample = request.form['sample']
    
    # Flashの設定
    input_flg = True
    
    if not mu:
        flash("平均値を入力してください")
        input_flg = False
        
    if not sd:
        flash("標準偏差を入力してください")
        input_flg = False
        
    if not sample:
        flash("自分の点数を入力してください")
        input_flg = False
        
    if not input_flg:
        return redirect(url_for("deviation_value_get"))
    
    # int型に変換
    mu = int(mu)
    sd = int(sd)
    sample = int(sample)
    
    # 値を求める
    post = True
    x = np.arange(0,100,0.01)
    y = stats.norm.pdf(x,mu,sd)
    t = (x - mu)/sd*10 + 50
    sample_t = (sample - mu)/sd*10 + 50
    percent = stats.norm.sf(x=(sample - mu)/sd)*100
    
    # プロットする
    plot1 = dev_plot1(x, y, sample, mu, sd)
    plot2 = dev_plot2(x, t, sample, sample_t)
    
    return render_template("1_deviation_value.html",
                           img=plot1, img2 = plot2,
                           value = round(sample_t, 1),
                           percent = round(percent, 1),
                           post=post)

def dev_plot1(x, y, sample, mu, sd):
    
    # プロットの作成
    plt.cla()
    plt.plot(x, y)
    plt.xlabel('テストの点数')
    plt.ylabel('テストの点数の分布')
    plt.title('テストの点数の分布')
    plt.fill_between(x,0,y,where=x>sample, facecolor='y',alpha=0.5, color='blue')
    plt.scatter(sample, stats.norm.pdf(sample,mu,sd), s=30)
    plt.text(sample+2, stats.norm.pdf(sample,mu,sd), "あなたの位置", fontsize="xx-large")
    
    # 変換
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return plot

def dev_plot2(x, y, sample, deviation_value):
    
    # プロットの作成
    plt.cla()
    plt.plot(x, y)
    plt.xlabel('テストの点数')
    plt.ylabel('偏差値の分布')
    plt.title('偏差値の分布')
    plt.scatter(sample, deviation_value, s=30)
    plt.text(sample+2, deviation_value-3, "あなたの位置", fontsize="xx-large")
    
    # 変換
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return plot

# 2 単回帰分析シミュレーションアプリ
@app.route('/linear', methods=['GET'])
def linear_get():
    return render_template('2_linear.html')

@app.route("/linear", methods=['POST'])
def linear_post():
    
    # 入力受け取り
    values = request.form.getlist('input')
    
    # Flashの設定
    input_flg = True
    
    if ("" in values):
        flash('全ての値を入力してください')
        input_flg = False
    
    if not input_flg:
        return redirect(url_for("linear_get"))
    
    post = True
    
    # int型に変換
    n = int(values[0])
    mu = int(values[1])
    sigma = int(values[2])
    a = int(values[3])
    b = int(values[4])
    sigma_epsilon = int(values[5])
    
    # 単回帰分析の実装
    x = np.random.normal(mu, sigma, n)
    y = a * x + b + np.random.normal(0, sigma_epsilon, n)
    a_hat, b_hat = linear_f(x, y)
    y_hat = a_hat*x + b_hat
    
    # プロットの可視化
    plot1 = sample_plot(x, y)
    plot2 = model_plot(x, y, y_hat)
    
    return render_template("2_linear.html",
                           img=plot1, img2 = plot2,
                           post = post,
                           values=values,
                           a_hat=round(a_hat, 2),
                           b_hat=round(b_hat,2))

def linear_f(x, y):
    n = len(x)
    a_hat = ((np.dot(x,y) - y.sum()*x.sum()/n))/((x**2).sum() - x.sum()**2/n)
    b_hat = (y.sum() - a_hat*x.sum())/n
    return a_hat, b_hat

def sample_plot(x, y):
    
    # プロットの作成
    plt.cla()
    plt.plot(x, y,'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('生成したデータ')
    
    # 変換
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return plot

def model_plot(x, y, y_hat):

    # プロットの作成
    plt.cla()
    plt.plot(x, y,'o')
    plt.plot(x, y_hat)    
    plt.xlabel('x')
    plt.ylabel('y_hat')
    plt.title('予測結果')
    
    # 変換
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plot = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return plot

# 3 物体検知アプリ
@app.route('/image_detect', methods=['GET'])
def image_detect_get():
    return render_template('3_image_detect.html')

@app.route('/image_detect', methods=['POST'])
def image_detect_post():
    
    # ファイルの作成
    img_dir = "static/images/"
    result_dir = "static/detected_images/"
    dir_list = [img_dir, result_dir]
    
    for item in dir_list:
        if os.path.exists(item):
            shutil.rmtree(item)
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 入力受け取り
    input_image = request.files['image']
    
    # Flaskの設定
    input_flg = True
    
    if not input_image:
        flash("画像を入力してください")
        input_flg = False
        
    if not input_flg:
        return redirect(url_for("image_detect_get"))
    
    # 画像の変換
    stream = input_image.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    
    # 画像の保存
    dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    img_path = img_dir + dt_now + ".jpg"
    result_path = result_dir + dt_now + ".jpg"
    cv2.imwrite(img_path, img)
    
    # 物体検知
    detect_image(img_path)
    
    return render_template('3_image_detect.html',
                           content=img_path,
                           content2=result_path)

def detect_image(img_path):
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    image = Image.open(img_path)
    results = model(image)
    results.save(save_dir="static/detected_images/")
    
    return 

# 4 画像生成アプリ
@app.route('/image_generate', methods=['GET'])
def image_generate_get():
    return render_template('4_image_generate.html',
                           title='Flaskでwebアプリ作成')

@app.route('/image_generate', methods=['POST'])
def image_generate_post():
    
    # ファイルの作成
    generate_dir = 'static/generated_images'
    
    if os.path.exists(generate_dir):
        shutil.rmtree(generate_dir)
        
    os.makedirs(generate_dir, exist_ok=True)
    
    # 入力受け取り
    number = request.form['number']
    n = request.form['n']
    
    # Flaskの設定
    input_flg = True
    
    if ("" in [number, n]):
        flash('全ての値を入力してください')
        return redirect(url_for("image_generate_get"))
    else :
        number = int(number)
        n = int(n)
    
    if number<0 or number>9:
        flash("0から9の値を入力してください")
        input_flg = False
        
    if n < 1 or n > 1000:
        flash("1から1000の値を入力してください")
        input_flg = False
        
    if not input_flg:
        return redirect(url_for("image_generate_post"))
    
    # 画像の生成
    img_path = generation.generate_image(number, n)
    
    return render_template('4_image_generate.html',
                           title='Flaskでwebアプリ作成',
                           content=img_path)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
    app.run(debug=True)