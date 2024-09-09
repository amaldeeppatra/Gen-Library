from flask import Flask, request, jsonify
from llama3 import get_answerllama
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query_llama():
    data = request.get_json()
    query = data.get('query')
    app.logger.debug(f'Received query: {query}')
    if query:
        try:
            answer = get_answerllama(query, 'embeddings')
            app.logger.debug(f'Generated answer: {answer}')
            return jsonify({'answer': answer})
        except Exception as e:
            app.logger.error(f'Error processing query: {str(e)}')
            return jsonify({'answer': f'Error processing query: {str(e)}'}), 500
    app.logger.warning('No query provided in request.')
    return jsonify({'answer': 'No query provided'}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
