{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0df55703-bdf3-48da-99ed-3ad3f59b69b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting Flask app...\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug:127.0.0.1 - - [12/Jul/2024 14:14:08] \"GET / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [12/Jul/2024 14:14:08] \"\u001b[36mGET /static/styles.css HTTP/1.1\u001b[0m\" 304 -\n",
      "INFO:werkzeug:127.0.0.1 - - [12/Jul/2024 14:14:08] \"GET /static/scripts.js HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rendering index page.\n",
      "File upload endpoint called.\n",
      "Predicting image at path: C:\\Users\\godof\\OneDrive\\Desktop\\cris project\\uploads\\AKASHAYTHAKUR_130_12107_193124_I_D_G.jpg\n",
      "WARNING:tensorflow:5 out of the last 7 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000185F1FABB00> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:5 out of the last 7 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000185F1FABB00> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 3s/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [12/Jul/2024 14:14:16] \"POST /predict HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction complete: Indian Dustbin Good Cleanliness (Confidence: 0.9992)\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, render_template, jsonify\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing import image\n",
    "import numpy as np\n",
    "import os\n",
    "import tensorflow as tf\n",
    "import base64\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load your model\n",
    "model = load_model('RailwayHygieneAI.h5')\n",
    "\n",
    "# Define your unique classes list (make sure it matches your model's output)\n",
    "unique_classes_list = [\n",
    "    'Indian Dustbin Bad Cleanliness',\n",
    "    'Indian Dustbin Good Cleanliness',\n",
    "    'Indian Dustbin Medium Cleanliness',\n",
    "    'Indian Toilet Bad Cleanliness',\n",
    "    'Indian Toilet Good Cleanliness',\n",
    "    'Indian Toilet Medium Cleanliness',\n",
    "    'Indian Washbasin Bad Cleanliness',\n",
    "    'Indian Washbasin Good Cleanliness',\n",
    "    'Indian Washbasin Medium Cleanliness',\n",
    "    'Western Dustbin Bad Cleanliness',\n",
    "    'Western Dustbin Good Cleanliness',\n",
    "    'Western Dustbin Medium Cleanliness',\n",
    "    'Western Toilet Bad Cleanliness',\n",
    "    'Western Toilet Good Cleanliness',\n",
    "    'Western Toilet Medium Cleanliness',\n",
    "    'Western Washbasin Bad Cleanliness',\n",
    "    'Western Washbasin Good Cleanliness',\n",
    "    'Western Washbasin Medium Cleanliness'\n",
    "]\n",
    "\n",
    "def model_predict(img_path, model):\n",
    "    print(f\"Predicting image at path: {img_path}\")\n",
    "    img = image.load_img(img_path, target_size=(256, 256))  # Load and resize image\n",
    "    x = image.img_to_array(img)  # Convert image to array\n",
    "    x = np.expand_dims(x, axis=0)  # Expand dimensions to match batch size\n",
    "    x /= 255.0  # Normalize image\n",
    "    preds = model.predict(x)  # Make prediction\n",
    "    predicted_class = np.argmax(preds, axis=1)[0]  # Get predicted class index\n",
    "    predicted_label = unique_classes_list[predicted_class]  # Get corresponding label\n",
    "    confidence = float(preds[0][predicted_class])  # Confidence score\n",
    "    print(f\"Prediction complete: {predicted_label} (Confidence: {confidence:.4f})\")\n",
    "    return predicted_label, confidence\n",
    "\n",
    "@app.route('/', methods=['GET'])\n",
    "def index():\n",
    "    print(\"Rendering index page.\")\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def upload():\n",
    "    print(\"File upload endpoint called.\")\n",
    "    if 'file' not in request.files:\n",
    "        return jsonify({'error': 'No file part'})\n",
    "\n",
    "    files = request.files.getlist('file')\n",
    "    if len(files) == 0:\n",
    "        return jsonify({'error': 'No files selected'})\n",
    "\n",
    "    results = []\n",
    "    for file in files:\n",
    "        if file.filename == '':\n",
    "            continue\n",
    "        basepath = os.getcwd()\n",
    "        file_path = os.path.join(basepath, 'uploads', file.filename)\n",
    "        file.save(file_path)\n",
    "\n",
    "        # Make prediction\n",
    "        predicted_label, confidence = model_predict(file_path, model)\n",
    "\n",
    "        # Encode image to base64\n",
    "        with open(file_path, \"rb\") as img_file:\n",
    "            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')\n",
    "\n",
    "        # Remove the file after prediction\n",
    "        os.remove(file_path)\n",
    "\n",
    "        results.append({'filename': file.filename, 'prediction': predicted_label, 'confidence': 100 * confidence, 'image': encoded_img})\n",
    "\n",
    "    return jsonify({'results': results})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    print(\"Starting Flask app...\")\n",
    "    app.run(debug=True, use_reloader=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "372b67ec-3338-4af6-a4e0-9e21cc6bad7a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
