from flask import Flask, request, render_template, send_file, url_for
from PIL import Image
import os
import shutil
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import subprocess
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static//uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

#パス変えました
app.config['DATASET_FOLDER'] = 'C:/hakkason/hakk/datasets/testA'
#阿部用パス
# app.config['DATASET_FOLDER'] = 'C:\hakk\datasets\\testA'

#パス変えました
app.config['RESULTS_FOLDER'] = 'hakk/pytorch-CycleGAN-and-pix2pix/results'
#阿部用パス
# app.config['RESULTS_FOLDER'] = './../hakk/pytorch-CycleGAN-and-pix2pix/results'

# CycleGANモデルの読み込み
model = tf.keras.models.load_model("./saved_modelsDrop/cyclegan_model", custom_objects={'InstanceNormalization': LayerNormalization})

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return tf.expand_dims(image, 0)  # バッチ次元を追加

def generate_image(model,input_image):
    prediction = model(input_image)
    return (prediction[0] * 0.5 + 0.5).numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    if 'file' not in request.files:
        return render_template('index.html', uploaded_img=url_for('static', filename='origin_sample.png'))

    target_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    processed_dir = app.config['PROCESSED_FOLDER']
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.mkdir(processed_dir)
    dataset_dir = app.config['DATASET_FOLDER']
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)

    file = request.files['file']
    artist = request.form['artist']

    if file.filename == '':
        return render_template('index.html', uploaded_img=url_for('static', filename='origin_sample.png'))

    if file:
        # filename = secure_filename(file.filename)
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 元の画像サイズを取得
        original_image = Image.open(file_path)
        original_size = original_image.size

        # 画像をdatasetsフォルダに保存
        dataset_path = os.path.join(dataset_dir, filename)
        shutil.copy(file_path, dataset_path)

        # ここでアーティストの値に基づいて適切な処理を行う
        print(f"選択されたアーティスト: {artist}")
        original_dir = os.getcwd()  # 元のディレクトリを保存

        #パス変えました
        target_dir = 'hakk/pytorch-CycleGAN-and-pix2pix'
        #阿部用パス
        # target_dir = './../hakk/pytorch-CycleGAN-and-pix2pix'
        if not os.path.isdir(target_dir):
            return f"指定されたパスが見つかりません: {target_dir}", 400
        try:
            if artist != 'monet':
                os.chdir(target_dir)

                # 仮想環境のPythonを使用するように変更
                venv_python = os.path.join(original_dir, 'venv', 'Scripts', 'python')

            result = None  # 初期化

            # 結果フォルダのクリア処理
            results_subfolder = os.path.join('results', f'style_{artist}_pretrained', 'test_latest', 'images')
            if os.path.exists(results_subfolder):
                shutil.rmtree(results_subfolder)
            os.makedirs(results_subfolder)

            if artist == "monet":
                input_image = preprocess_image(file_path)
                processed_image_np = generate_image(model,input_image)
                processed_image_pil = Image.fromarray((processed_image_np * 255).astype(np.uint8))
                processed_image_path = os.path.join(processed_dir,filename)
                processed_image_pil.save(processed_image_path)
                
                processed_image = Image.open(processed_image_path)
                processed_image = processed_image.resize(original_size)
                processed_image_resized_path = os.path.join(processed_dir, f"resized_{filename}")
                processed_image.save(processed_image_resized_path)

                return render_template('index.html',uploaded_img=file_path,processed_img=processed_image_resized_path)
            
            elif artist == "vangogh":
                result = subprocess.run([venv_python, 'test.py', '--dataroot', dataset_dir, '--name', 'style_vangogh_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], capture_output=True, text=True)
            elif artist == "cezanne":
                result = subprocess.run([venv_python, 'test.py', '--dataroot', dataset_dir, '--name', 'style_cezanne_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], capture_output=True, text=True)
            elif artist == "ukiyoe":
                result = subprocess.run([venv_python, 'test.py', '--dataroot', dataset_dir, '--name', 'style_ukiyoe_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], capture_output=True, text=True)

            if result and result.returncode != 0:
                return f"サブプロセスの実行中にエラーが発生しました: {result.stderr}", 500
        except subprocess.CalledProcessError as e:
            return f"サブプロセスの実行中にエラーが発生しました: {e}", 500
        finally:
            os.chdir(original_dir)  # 元のディレクトリに戻る

        # 変換後の画像パス
        processed_image_path = os.path.join(app.config['RESULTS_FOLDER'], f'style_{artist}_pretrained', 'test_latest', 'images', f'{filename.split(".")[0]}_fake.png')
        print(filename)
        if not os.path.exists(processed_image_path):
            return f"変換後の画像が見つかりません: {processed_image_path}", 500

        # 変換後の画像を元のサイズにリサイズ
        processed_image = Image.open(processed_image_path)
        processed_image = processed_image.resize(original_size)
        static_processed_image_path = os.path.join(processed_dir, f'{filename.split(".")[0]}_fake.png')
        processed_image.save(static_processed_image_path)

        # Webサーバーが画像を提供できるように相対パスに変換
        web_image_path = f'static/processed/{filename.split(".")[0]}_fake.png'
        print(f"Processed Image Path: {web_image_path}")

        return render_template('index.html', uploaded_img=file_path, processed_img=web_image_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)