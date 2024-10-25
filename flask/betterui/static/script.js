let dropArea = document.getElementById('drop-area');
let chatBox = document.getElementById('chat-box');
let userInput = document.getElementById('user-input');
let sendBtn = document.getElementById('send-btn');

dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropArea.classList.add('highlight');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('highlight');
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dropArea.classList.remove('highlight');
    let files = event.dataTransfer.files;
    handleFileUpload(files[0]);
});

function handleFileUpload(file) {
    if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file.');
        return;
    }
    let formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            chatBox.innerHTML += `<div>Error: ${data.error}</div>`;
        } else {
            chatBox.innerHTML += `<div>Response: ${data.response}</div>`;
        }
    });
}

sendBtn.addEventListener('click', () => {
    let message = userInput.value;
    if (message.trim() === "") return; // Prevent sending empty messages
    chatBox.innerHTML += `<div>You: ${message}</div>`;
    userInput.value = '';  // Clear the input after sending
});

// Add this function to listen for Enter key press
userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent the default form submission
        sendBtn.click(); // Trigger the click event on the send button
    }
});
