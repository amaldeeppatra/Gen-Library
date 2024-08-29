from flask import Flask, request, jsonify, render_template
from llama3 import get_answerllama

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data['query']
    # Call your get_answerllama function with the query and user_folder
    result = get_answerllama(query, 'embeddings')
    return jsonify({'answer': result['result']})

if __name__ == '__main__':
    app.run(debug=True)
