// Variable globale pour la session ONNX
let session;

// Configuration du Canvas
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.lineWidth = 20;       // Trait épais (important pour MNIST)
ctx.lineCap = 'round';    // Bouts ronds pour fluidité
ctx.strokeStyle = 'white'; // Dessin blanc
ctx.fillStyle = 'black';  // Fond noir

// Initialiser le fond noir
ctx.fillRect(0, 0, canvas.width, canvas.height);

// --- Gestion de la souris / Tactile pour dessiner ---
let isDrawing = false;

function startPosition(e) {
    isDrawing = true;
    draw(e);
}
function endPosition() {
    isDrawing = false;
    ctx.beginPath();
}
function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || e.touches[0].clientX;
    const clientY = e.clientY || e.touches[0].clientY;

    ctx.lineTo(clientX - rect.left, clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(clientX - rect.left, clientY - rect.top);
}

canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);
// Événements Tactiles (Mobile)
canvas.addEventListener('touchstart', startPosition);
canvas.addEventListener('touchend', endPosition);
canvas.addEventListener('touchmove', draw);

async function loadModel() {
    try {
    
        session = await ort.InferenceSession.create('./model.onnx', { executionProviders: ['wasm'] });
        
        document.getElementById('status').innerText = "Modèle prêt ! (WASM)";
        console.log("Session ONNX chargée");
    } catch (e) {
        document.getElementById('status').innerText = "Erreur chargement modèle";
        console.error(e);
    }
}
loadModel();

// Fonction pour effacer
function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = "";
    ctx.beginPath();
}

function processImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.imageSmoothingEnabled = true; 
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data; 
    
    const input = new Float32Array(28 * 28);
    
    const mean = 0.1307;
    const std = 0.3081;

    for (let i = 0; i < 28 * 28; i++) {
        let val = data[i * 4] / 255.0;
        
        input[i] = (val - mean) / std;
    }
    
    return input;
}

async function runInference() {
    if (!session) {
        alert("Le modèle n'est pas encore chargé !");
        return;
    }

    const inputData = processImage();

    const dims = [1, 1, 28, 28]; 
    const tensor = new ort.Tensor('float32', inputData, dims);


    const inputName = session.inputNames[0];  
    const outputName = session.outputNames[0];

    const feeds = {};
    feeds[inputName] = tensor; 

    console.log(`Entrée utilisée: ${inputName}, Sortie attendue: ${outputName}`);

    try {
        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        const output = results[outputName].data;

        const maxIndex = output.indexOf(Math.max(...output));
        
        document.getElementById('result').innerText = `Prédiction : ${maxIndex}`;
        console.log(`Temps d'inférence : ${(end - start).toFixed(2)}ms`);

    } catch (e) {
        console.error("Erreur pendant l'inférence :", e);
        document.getElementById('result').innerText = "Erreur !";
    }
}