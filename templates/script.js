
        let video = document.getElementById('video');
        let currentSentence = '';
        let stream = null;
        let predictionInterval = null;
        let lastPrediction = '';
        let predictionCount = 0;
        let totalLettersDetected = 0;

        function showApp() {
            document.getElementById('landingPage').style.display = 'none';
            document.getElementById('appPage').classList.add('active');
        }

        function showLanding() {
            document.getElementById('appPage').classList.remove('active');
            document.getElementById('landingPage').style.display = 'flex';
            stopCamera();
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                document.getElementById('cameraStatus').textContent = '🔴 Live';
                document.getElementById('cameraStatus').style.background = 'rgba(239, 68, 68, 0.8)';
                startPredictionLoop();
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
                document.getElementById('cameraStatus').textContent = '📷 Camera Off';
                document.getElementById('cameraStatus').style.background = 'rgba(0, 0, 0, 0.8)';
            }
            if (predictionInterval) {
                clearInterval(predictionInterval);
                predictionInterval = null;
            }
        }

        function clearSentence() {
            currentSentence = '';
            lastPrediction = '';
            predictionCount = 0;
            totalLettersDetected = 0;
            document.getElementById('sentenceDisplay').textContent = 'Start making gestures to build your message...';
            document.getElementById('totalLetters').textContent = '0';
            document.getElementById('wordsCount').textContent = '0';
        }

        function updateStats() {
            document.getElementById('totalLetters').textContent = totalLettersDetected;
            const words = currentSentence.trim().split(/\s+/).filter(w => w.length > 0).length;
            document.getElementById('wordsCount').textContent = words;
        }

        function startPredictionLoop() {
            if (predictionInterval) clearInterval(predictionInterval);
            predictionInterval = setInterval(async () => {
                if (!stream) return;
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    formData.append('model', document.getElementById('modelSelect').value);
                    formData.append('draw_landmarks', 'false');
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        if (data.prediction) {
                            document.getElementById('currentLetter').textContent = data.prediction;
                            const confidencePercent = (data.confidence * 100).toFixed(1);
                            document.getElementById('confidence').textContent = `Confidence: ${confidencePercent}%`;
                            document.getElementById('confidenceFill').style.width = confidencePercent + '%';
                            if (data.confidence > 0.7) {
                                if (data.prediction === lastPrediction) {
                                    predictionCount++;
                                    if (predictionCount >= 3) {
                                        currentSentence += data.prediction;
                                        totalLettersDetected++;
                                        document.getElementById('sentenceDisplay').textContent = currentSentence || 'Start making gestures...';
                                        updateStats();
                                        predictionCount = 0;
                                        lastPrediction = '';
                                    }
                                } else {
                                    lastPrediction = data.prediction;
                                    predictionCount = 1;
                                }
                            }
                        } else if (data.error) {
                            document.getElementById('currentLetter').textContent = '?';
                            document.getElementById('confidence').textContent = data.error;
                            document.getElementById('confidenceFill').style.width = '0%';
                        }
                    } catch (err) {
                        console.error('Prediction error:', err);
                    }
                }, 'image/jpeg');
            }, 1000);
        }

        document.getElementById('modelSelect').addEventListener('change', (e) => {
            fetch('/set_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: e.target.value })
            }).then(response => response.json())
              .then(data => console.log('Model changed to:', data.current_model))
              .catch(err => console.error('Error changing model:', err));
        });