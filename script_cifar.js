const CANVAS_SIZE = 320;
const MODEL_INPUT_SIZE = 32; // CIFAR est en 32x32
let session;


const CLASSES = ['Avion', 'Automobile', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'];

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

ctx.fillStyle = "white"; 
ctx.fillRect(0, 0, canvas.width, canvas.height);

const imageLoader = document.getElementById('imageLoader');
imageLoader.addEventListener('change', handleImage, false);

function handleImage(e){
    const reader = new FileReader();
    reader.onload = function(event){
        const img = new Image();
        img.onload = function(){
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
        img.src = event.target.result;
    }
    reader.readAsDataURL(e.target.files[0]);
}

let isDrawing = false;
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black'; 

function startPosition(e) { isDrawing = true; draw(e); }
function endPosition() { isDrawing = false; ctx.beginPath(); }
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
canvas.addEventListener('touchstart', startPosition);
canvas.addEventListener('touchend', endPosition);
canvas.addEventListener('touchmove', draw);

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('./model_cifar.onnx', { executionProviders: ['wasm'] });
        document.getElementById('status').innerText = "Modèle CIFAR prêt !";
        console.log("Session chargée");
    } catch (e) {
        document.getElementById('status').innerText = "Erreur chargement (vérifiez le nom du fichier .onnx)";
        console.error(e);
    }
}
loadModel();

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerText = "";
    ctx.beginPath();
}

function processImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MODEL_INPUT_SIZE;
    tempCanvas.height = MODEL_INPUT_SIZE;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.imageSmoothingEnabled = true; 
    tempCtx.drawImage(canvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    
    const imageData = tempCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const data = imageData.data; 

    const input = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
    
    const mean = [0.4914, 0.4822, 0.4465];
    const std = [0.2023, 0.1994, 0.2010];

    for (let i = 0; i < MODEL_INPUT_SIZE * MODEL_INPUT_SIZE; i++) {
        let r = data[i * 4] / 255.0;
        let g = data[i * 4 + 1] / 255.0;
        let b = data[i * 4 + 2] / 255.0;

        r = (r - mean[0]) / std[0];
        g = (g - mean[1]) / std[1];
        b = (b - mean[2]) / std[2];

        
        input[i] = r; 
        input[i + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = g; 
        input[i + 2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE] = b; 
    }
    
    return input;
}

async function runInference() {
    if (!session) { alert("Attendez le chargement du modèle !"); return; }

    const inputData = processImage();

    const dims = [1, 3, 32, 32]; 
    const tensor = new ort.Tensor('float32', inputData, dims);

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const feeds = {};
    feeds[inputName] = tensor; 

    try {
        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        const output = results[outputName].data;
        
        const maxIndex = output.indexOf(Math.max(...output));
        const className = CLASSES[maxIndex];

        document.getElementById('result').innerText = `Prédiction : ${className} (${Math.max(...output).toFixed(2)})`;
        console.log(`Temps : ${(end - start).toFixed(2)}ms`);

    } catch (e) {
        console.error(e);
        document.getElementById('result').innerText = "Erreur inférence (voir console)";
    }
}