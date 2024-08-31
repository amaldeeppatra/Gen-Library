const statusDisplay = document.getElementById('status');
const resultDisplay = document.getElementById('result');
const transcriptDisplay = document.getElementById('transcript');
const startRecordBtn = document.getElementById('start-record-btn');
const stopRecordBtn = document.getElementById('stop-record-btn');

let recognition;

function startRecording() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert("Your browser doesn't support speech recognition.");
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = function () {
        statusDisplay.innerHTML = 'Listening...';
        startRecordBtn.disabled = true;
        stopRecordBtn.disabled = false;
    };

    recognition.onresult = function (event) {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        transcriptDisplay.value = finalTranscript + interimTranscript;
    };

    recognition.onerror = function (event) {
        statusDisplay.innerHTML = 'Error occurred: ' + event.error;
        console.error('Speech Recognition Error:', event.error);
    };

    recognition.onend = function () {
        statusDisplay.innerHTML = 'Stopped listening.';
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;
    };

    recognition.start();
}

function stopRecording() {
    if (recognition) {
        recognition.stop();
        sendQuery(transcriptDisplay.value.trim());
    }
}

function sendQuery(query) {
    if (!query) {
        alert("Please provide a query.");
        return;
    }

    statusDisplay.innerHTML = 'Processing your query...';

    fetch('http://localhost:5000/query', {  // Ensure this URL matches your Flask server
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            resultDisplay.value = data.answer;
            statusDisplay.innerHTML = 'Query processed.';
        })
        .catch(error => {
            console.error('Error:', error);
            resultDisplay.value = 'An error occurred while getting the answer.';
            statusDisplay.innerHTML = 'Error processing your query.';
        });
}
