const video = document.getElementById('video');
const gestureForm = document.getElementById('gesture-form');
const predictedText = document.getElementById('predicted-text');

// Get video stream from camera
navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(error => {
    console.error(error);
  });

// Submit gesture to server and get predicted text
gestureForm.addEventListener('submit', event => {
  event.preventDefault();
  const gestureInput = document.getElementById('gesture').value;
  const formData = new FormData();
  formData.append('gesture', gestureInput);
  fetch('/predict', {method: 'POST', body: formData})
    .then(response => response.text())
    .then(text => {
      predictedText.textContent = text;
    })
    .catch(error => {
      console.error(error);
    })
  })
