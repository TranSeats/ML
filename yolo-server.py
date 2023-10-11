from flask import Flask, request, jsonify
import subprocess
import os
import re
# Fungsi untuk memanggil Darknet melalui command line
app =Flask(__name__)
def run_darknet(image_path):
    darknet_command = f"./darknet detector test ~/Despro/crowdhuman-416x416.data ~/Despro/yolov4-crowdhuman-416x416.cfg ~/Despro/yolov4-crowdhuman-416x416_best.weights {image_path} -gpus 0 -dont_show"
    result = subprocess.run(darknet_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8')
# Fungsi untuk menghitung jumlah head dan person dari output Darknet
def count_classes(output):
    class_counts = {
        'head': 0,
        'person': 0
    }
    lines = output.split('\n')
    for line in lines:
        if line.strip().endswith('%'):
            class_name = line.split(':')[0].strip()
            if class_name in class_counts:
                class_counts[class_name] += 1
    return class_counts
# Endpoint untuk melakukan deteksi objek
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Simpan gambar ke file
    image_path = request.json.get('filename')
    # Jalankan Darknet
    darknet_output = run_darknet(image_path)
    # Hitung jumlah head dan person
    class_counts = count_classes(darknet_output)
    # Hapus file gambar yang diunggah
    os.remove(image_path)
    return jsonify(class_counts), 200

app.run()
