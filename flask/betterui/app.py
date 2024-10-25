from flask import Flask, render_template, request, jsonify
import os


from pydantic import BaseModel



app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # response = vectara_api_call(file_path)  # Implement this function as needed

    # using a dummy response
    response = "File '{}' has been uploaded successfully (dummy response for now)".format(file_path)

    return jsonify({'response': response})


class ChatRequest(BaseModel):
    message: str
    isChecked: bool

class urls(BaseModel):
    imgURL: str
    audioURL: str


if __name__ == '__main__':
    app.run(debug=True)
