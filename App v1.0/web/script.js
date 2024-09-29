const dropArea = document.getElementById('dropArea');
const fileElem = document.getElementById('fileElem');
const fileSelect = document.getElementById('fileSelect');
const gallery = document.getElementById('gallery');
const legendList = document.getElementById('legendList');

// Event listeners
dropArea.addEventListener('dragenter', highlight, false);
dropArea.addEventListener('dragover', highlight, false);
dropArea.addEventListener('dragleave', unhighlight, false);
dropArea.addEventListener('drop', handleDrop, false);
fileSelect.addEventListener('click', () => fileElem.click(), false);
fileElem.addEventListener('change', (e) => handleFiles(e.target.files), false);

// Drag and drop functions
function highlight(e) {
    preventDefault(e);
    dropArea.classList.add('bg-gray-200');
}

function unhighlight(e) {
    preventDefault(e);
    dropArea.classList.remove('bg-gray-200');
}

function handleDrop(e) {
    preventDefault(e);
    unhighlight(e);
    handleFiles(e.dataTransfer.files);
}

function preventDefault(e) {
    e.preventDefault();
    e.stopPropagation();
}

// File processing
async function handleFiles(files) {
    for (const file of files) {
        await processFile(file);
    }
}

async function processFile(file) {
    const originalImage = await displayImage(file, 'Original');
    gallery.appendChild(originalImage);

    try {
        const reader = new FileReader();
        const fileData = await new Promise((resolve) => {
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(file);
        });

        const result = await eel.process_image(fileData)();
        if (result) {
            const processedImage = createImageElement(`data:image/png;base64,${result.processedImage}`, 'Processed');
            gallery.appendChild(processedImage);

            // Display the detections
            const detections = result.detections;
            detections.forEach(detection => {
                const detectionElement = document.createElement('div');
                detectionElement.textContent = `Class ${detection.class}: ${detection.confidence.toFixed(2)}`;
                gallery.appendChild(detectionElement);
            });
        } else {
            throw new Error('Image processing failed');
        }
    } catch (error) {
        console.error('Error processing image:', error);
        displayError(file.name);
    }
}

function displayImage(file, label) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = createImageElement(e.target.result, label);
            resolve(img);
        };
        reader.readAsDataURL(file);
    });
}

function createImageElement(src, label) {
    const container = document.createElement('div');
    container.className = 'image-container';

    const img = document.createElement('img');
    img.src = src;
    img.alt = label;
    img.className = 'max-w-full h-auto rounded shadow-md';

    const labelElement = document.createElement('div');
    labelElement.textContent = label;
    labelElement.className = 'image-label mt-2 text-sm font-semibold';

    container.appendChild(img);
    container.appendChild(labelElement);

    return container;
}

function displayError(fileName) {
    const errorElement = document.createElement('div');
    errorElement.className = 'image-container p-4 bg-red-100 text-red-700 rounded';
    errorElement.textContent = `Error processing ${fileName}. Please try again.`;
    gallery.appendChild(errorElement);
}

// Load class names from the backend and populate the legend
eel.get_class_names()().then((classNames) => {
    classNames.forEach(className => {
        const li = document.createElement('li');
        li.textContent = className;
        legendList.appendChild(li);
    });
});