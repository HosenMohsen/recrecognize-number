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
    
    // Récupérer la position souris ou touch
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || e.touches[0].clientX;
    const clientY = e.clientY || e.touches[0].clientY;

    ctx.lineTo(clientX - rect.left, clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(clientX - rect.left, clientY - rect.top);
}

// Événements Souris
canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);
// Événements Tactiles (Mobile)
canvas.addEventListener('touchstart', startPosition);
canvas.addEventListener('touchend', endPosition);
canvas.addEventListener('touchmove', draw);

// --- 1. CHARGEMENT DU MODÈLE ONNX ---
async function loadModel() {
    try {
        // Utilisation de l'API ort (onnxruntime)
        // 'wasm' est le backend par défaut (WebAssembly), compatible partout.
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

// --- 2. PRÉTRAITEMENT DE L'IMAGE ---
function processImage() {
    // 1. Redimensionner l'image du canvas (280x280) vers (28x28)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // 2. Récupérer les données brutes des pixels
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data; // Tableau [r, g, b, a, r, g, b, a, ...]
    
    // 3. Convertir en Float32, Normaliser (0 à 1), puis Binariser (0 ou 1)
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // On prend juste le canal Rouge (puisque c'est gris/blanc/noir)
        // data[i*4] = Rouge. 
        const normalizedValue = data[i * 4] / 255.0;
        // Binarisation : si > 0 alors 1, sinon 0
        input[i] = normalizedValue > 0 ? 1 : 0;
    }
    
    return input;
}

// --- 3. INFÉRENCE (PRÉDICTION) ---
// --- 3. INFÉRENCE (PRÉDICTION) ---
async function runInference() {
    if (!session) {
        alert("Le modèle n'est pas encore chargé !");
        return;
    }

    const inputData = processImage();

    // Création du tenseur ONNX [1, 28, 28]
    const dims = [1, 28, 28]; 
    const tensor = new ort.Tensor('float32', inputData, dims);

    // --- CORRECTION MAJEURE ICI ---
    
    // 1. Récupérer dynamiquement les noms définis dans le modèle
    // (Cela évite les erreurs si vous avez nommé l'entrée "input" ou "x" en Python)
    const inputName = session.inputNames[0];  
    const outputName = session.outputNames[0];

    // 2. Préparer l'objet feeds avec le BON nom d'entrée
    const feeds = {};
    feeds[inputName] = tensor; 

    console.log(`Entrée utilisée: ${inputName}, Sortie attendue: ${outputName}`);

    try {
        // Exécution
        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        // 3. Lire le résultat avec le BON nom de sortie
        // Au lieu de faire results.output.data, on fait results[outputName].data
        const output = results[outputName].data;

        // Trouver l'index max (Argmax)
        const maxIndex = output.indexOf(Math.max(...output));
        
        document.getElementById('result').innerText = `Prédiction : ${maxIndex}`;
        console.log(`Temps d'inférence : ${(end - start).toFixed(2)}ms`);

    } catch (e) {
        console.error("Erreur pendant l'inférence :", e);
        document.getElementById('result').innerText = "Erreur !";
    }
}