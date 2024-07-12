document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const resultSection = document.getElementById('resultSection');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data.results);
        } catch (error) {
            console.error('Error:', error);
        }
    });

    function displayResults(results) {
        resultSection.innerHTML = ''; // Clear previous results

        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.classList.add('result-section');
            resultDiv.innerHTML = `
                <h2>${result.filename}</h2>
                <img src="data:image/jpeg;base64,${result.image}" alt="Uploaded Image">
                <p><strong>Prediction:</strong> ${result.prediction}</p>
                <p><strong>Confidence:</strong> ${result.confidence.toFixed(2)}%</p>
            `;
            resultSection.appendChild(resultDiv);
        });
    }
});
