/**
 * Eraseon - Client Application
 * Canvas masking engine, auto-detection integration, split-view comparison, and PDF processing.
 */

// Application State
const state = {
  sessionId: null,
  fileType: null,
  filename: null,
  pageCount: 1,
  currentPage: 0,
  currentTool: 'brush', // brush | rect | eraser
  brushSize: 28,
  zoom: 1.0,
  isDrawing: false,
  rectStartX: 0,
  rectStartY: 0,
  originalImage: null,
  cleanedImageSrc: null,
  pageMasks: {}, // cache of mask dataURL per page
  detectedBoxes: [], // list of detected watermark bounding boxes
  applyAllPages: true,
  isSplitDragging: false,
};

// DOM Elements
const fileUploadInput = document.getElementById('fileUploadInput');
const dropZonePrompt = document.getElementById('dropZonePrompt');
const canvasStage = document.getElementById('canvasStage');
const comparisonStage = document.getElementById('comparisonStage');
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const imgCtx = imageCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
const brushCursor = document.getElementById('brushCursor');
const canvasViewport = document.getElementById('canvasViewport');

// Detection Banner & Sidebar Elements
const detectionBanner = document.getElementById('detectionBanner');
const detectionTitle = document.getElementById('detectionTitle');
const detectionDesc = document.getElementById('detectionDesc');
const btnProceedRemoval = document.getElementById('btnProceedRemoval');
const btnDismissBoxes = document.getElementById('btnDismissBoxes');
const detectedItemsSection = document.getElementById('detectedItemsSection');
const detectedBoxesCountBadge = document.getElementById('detectedBoxesCountBadge');
const btnClearDetectedBoxes = document.getElementById('btnClearDetectedBoxes');
const detectedBoxesList = document.getElementById('detectedBoxesList');

// Header & Info Elements
const fileMetaBadge = document.getElementById('fileMetaBadge');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const fileTypeBadge = document.getElementById('fileTypeBadge');

// Controls
const toolBrush = document.getElementById('toolBrush');
const toolRect = document.getElementById('toolRect');
const toolEraser = document.getElementById('toolEraser');
const toolClear = document.getElementById('toolClear');
const brushSizeRange = document.getElementById('brushSizeRange');
const brushSizeVal = document.getElementById('brushSizeVal');

const btnAutoDetect = document.getElementById('btnAutoDetect');
const detectSensRange = document.getElementById('detectSensRange');
const detectSensVal = document.getElementById('detectSensVal');
const dilateRange = document.getElementById('dilateRange');
const dilateVal = document.getElementById('dilateVal');

const removalModeSelect = document.getElementById('removalModeSelect');
const opacityRange = document.getElementById('opacityRange');
const opacityVal = document.getElementById('opacityVal');
const colorHintSelect = document.getElementById('colorHintSelect');

// PDF Controls
const pdfControlsSection = document.getElementById('pdfControlsSection');
const checkApplyAllPages = document.getElementById('checkApplyAllPages');
const btnProcessAllPages = document.getElementById('btnProcessAllPages');
const pdfPagination = document.getElementById('pdfPagination');
const btnPrevPage = document.getElementById('btnPrevPage');
const btnNextPage = document.getElementById('btnNextPage');
const pageIndicator = document.getElementById('pageIndicator');

// View Toggles & Actions
const btnViewEditor = document.getElementById('btnViewEditor');
const btnViewComparison = document.getElementById('btnViewComparison');
const btnProcessPage = document.getElementById('btnProcessPage');
const btnDownload = document.getElementById('btnDownload');
const downloadBtnText = document.getElementById('downloadBtnText');

// Split View Elements
const splitContainer = document.getElementById('splitContainer');
const splitOverlay = document.getElementById('splitOverlay');
const splitHandle = document.getElementById('splitHandle');
const imgOriginal = document.getElementById('imgOriginal');
const imgCleaned = document.getElementById('imgCleaned');

// Loading Overlay
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const loadingSubtext = document.getElementById('loadingSubtext');

// Zoom controls
const btnZoomIn = document.getElementById('btnZoomIn');
const btnZoomOut = document.getElementById('btnZoomOut');
const btnFitScreen = document.getElementById('btnFitScreen');
const zoomLevel = document.getElementById('zoomLevel');

// --- Initialization & Event Listeners ---

function init() {
  setupFileUpload();
  setupTools();
  setupCanvasDrawing();
  setupAutoDetect();
  setupProcessing();
  setupSplitSlider();
  setupPagination();
  setupZoom();
}

// 1. File Upload (Input & Drag and Drop)
function setupFileUpload() {
  fileUploadInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  });

  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });
}

async function uploadFile(file) {
  showLoading('Uploading file...', `Processing ${file.name}`);
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    state.sessionId = data.session_id;
    state.fileType = data.file_type;
    state.filename = data.filename;
    state.pageCount = data.page_count;
    state.currentPage = 0;
    state.pageMasks = {};
    state.detectedBoxes = [];
    removalModeSelect.value = 'auto';

    // Update Header
    fileMetaBadge.classList.remove('hidden');
    fileNameDisplay.textContent = state.filename;
    fileTypeBadge.textContent = state.fileType.toUpperCase();

    // Toggle PDF UI
    if (state.fileType === 'pdf') {
      pdfControlsSection.classList.remove('hidden');
      pdfPagination.classList.remove('hidden');
      updatePaginationUI();
    } else {
      pdfControlsSection.classList.add('hidden');
      pdfPagination.classList.add('hidden');
    }

    // Load first page and then trigger upload-time auto-detection
    await loadPage(0, /* autoDetectOnLoad= */ true);
  } catch (err) {
    alert('Upload failed: ' + err.message);
  } finally {
    hideLoading();
  }
}

// 2. Load and Render Page
async function loadPage(pageNum, autoDetectOnLoad = false) {
  showLoading('Rendering page...', `Loading page ${pageNum + 1}`);
  try {
    const res = await fetch(`/api/page/${state.sessionId}/${pageNum}`);
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);

    const img = new Image();
    img.onload = async () => {
      // Switch to Editor View BEFORE fitToScreen so viewport dimensions include scrollbars accurately
      switchView('editor');
      dropZonePrompt.classList.add('hidden');
      canvasStage.classList.remove('hidden');

      state.originalImage = img;
      renderImageToCanvas(img);

      // Setup image for split comparison
      imgOriginal.src = objectUrl;
      btnViewComparison.disabled = true;
      state.cleanedImageSrc = null;

      // Clear previous overlay badges when changing page
      if (overlayCtx && overlayCanvas) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      }

      // Restore or clear mask
      if (state.pageMasks[pageNum]) {
        restoreMaskFromDataUrl(state.pageMasks[pageNum]);
      } else if (state.applyAllPages && state.pageMasks['template']) {
        restoreMaskFromDataUrl(state.pageMasks['template']);
      } else {
        clearMask();
      }

      hideLoading();

      // Trigger auto-detect (Only detect as box, with option to proceed)
      if (autoDetectOnLoad) {
        await autoDetectOnUpload();
      }
    };
    img.src = objectUrl;
  } catch (err) {
    alert('Failed to load page: ' + err.message);
    hideLoading();
  }
}

function renderImageToCanvas(img) {
  imageCanvas.width = img.naturalWidth;
  imageCanvas.height = img.naturalHeight;
  maskCanvas.width = img.naturalWidth;
  maskCanvas.height = img.naturalHeight;
  if (overlayCanvas) {
    overlayCanvas.width = img.naturalWidth;
    overlayCanvas.height = img.naturalHeight;
  }

  imgCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  imgCtx.drawImage(img, 0, 0);

  // Automatically fit image to screen so corners and all watermarks are visible
  fitToScreen();
}

// 3. Selection Tools Setup
function setupTools() {
  const tools = [
    { btn: toolBrush, name: 'brush' },
    { btn: toolRect, name: 'rect' },
    { btn: toolEraser, name: 'eraser' }
  ];

  tools.forEach(({ btn, name }) => {
    btn.addEventListener('click', () => {
      tools.forEach(t => t.btn.classList.remove('active'));
      btn.classList.add('active');
      state.currentTool = name;
    });
  });

  toolClear.addEventListener('click', () => {
    clearMask();
    hideDetectionBanner();
  });

  brushSizeRange.addEventListener('input', (e) => {
    state.brushSize = parseInt(e.target.value);
    brushSizeVal.textContent = `${state.brushSize}px`;
    updateBrushCursorSize();
  });

  detectSensRange.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    detectSensVal.textContent = v <= 0.3 ? 'Low' : (v <= 0.6 ? 'Medium' : 'High');
  });

  dilateRange.addEventListener('input', (e) => {
    dilateVal.textContent = `${e.target.value}px`;
  });

  opacityRange.addEventListener('input', (e) => {
    opacityVal.textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`;
  });

  checkApplyAllPages.addEventListener('change', (e) => {
    state.applyAllPages = e.target.checked;
  });
}

function clearMask() {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  if (overlayCtx && overlayCanvas) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }
  state.detectedBoxes = [];
  saveCurrentMask();
}

function saveCurrentMask() {
  const dataUrl = maskCanvas.toDataURL('image/png');
  state.pageMasks[state.currentPage] = dataUrl;
  if (state.applyAllPages) {
    state.pageMasks['template'] = dataUrl;
  }
}

function restoreMaskFromDataUrl(dataUrl) {
  const maskImg = new Image();
  maskImg.onload = () => {
    maskCtx.globalCompositeOperation = 'source-over';
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
  };
  maskImg.src = dataUrl;
}

// 4. Interactive Canvas Drawing
function setupCanvasDrawing() {
  const getCanvasCoords = (e) => {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? (maskCanvas.width / rect.width) : 1;
    const scaleY = rect.height > 0 ? (maskCanvas.height / rect.height) : 1;
    const rawX = (e.clientX - rect.left) * scaleX;
    const rawY = (e.clientY - rect.top) * scaleY;
    return {
      x: Math.max(0, Math.min(maskCanvas.width, rawX)),
      y: Math.max(0, Math.min(maskCanvas.height, rawY)),
      viewportX: e.clientX,
      viewportY: e.clientY
    };
  };

  maskCanvas.addEventListener('mouseenter', () => {
    if (state.currentTool === 'brush' || state.currentTool === 'eraser') {
      brushCursor.style.display = 'block';
    }
  });

  maskCanvas.addEventListener('mouseleave', () => {
    brushCursor.style.display = 'none';
    if (state.isDrawing && state.currentTool !== 'rect') {
      state.isDrawing = false;
      saveCurrentMask();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.isDrawing && e.target !== maskCanvas) return;
    const coords = getCanvasCoords(e);

    if (e.target === maskCanvas) {
      const stageRect = canvasStage.getBoundingClientRect();
      brushCursor.style.left = `${e.clientX - stageRect.left}px`;
      brushCursor.style.top = `${e.clientY - stageRect.top}px`;
    }

    if (!state.isDrawing) return;

    if (state.currentTool === 'brush') {
      drawBrushStroke(coords.x, coords.y);
    } else if (state.currentTool === 'eraser') {
      eraseStroke(coords.x, coords.y);
    } else if (state.currentTool === 'rect') {
      redrawWithRectPreview(coords.x, coords.y);
    }
  });

  maskCanvas.addEventListener('mousedown', (e) => {
    state.isDrawing = true;
    const coords = getCanvasCoords(e);

    if (state.currentTool === 'brush') {
      maskCtx.globalCompositeOperation = 'source-over';
      maskCtx.beginPath();
      maskCtx.moveTo(coords.x, coords.y);
      drawBrushStroke(coords.x, coords.y);
    } else if (state.currentTool === 'eraser') {
      eraseStroke(coords.x, coords.y);
    } else if (state.currentTool === 'rect') {
      maskCtx.globalCompositeOperation = 'source-over';
      state.rectStartX = coords.x;
      state.rectStartY = coords.y;
      state.maskSnapshot = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    }
  });

  window.addEventListener('mouseup', (e) => {
    if (!state.isDrawing) return;
    state.isDrawing = false;

    if (state.currentTool === 'rect') {
      if (state.maskSnapshot) {
        maskCtx.putImageData(state.maskSnapshot, 0, 0);
        state.maskSnapshot = null;
      }
      const coords = getCanvasCoords(e);
      const x = Math.min(state.rectStartX, coords.x);
      const y = Math.min(state.rectStartY, coords.y);
      const w = Math.abs(coords.x - state.rectStartX);
      const h = Math.abs(coords.y - state.rectStartY);

      if (w >= 2 && h >= 2) {
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.fillStyle = 'rgba(239, 68, 68, 0.55)';
        maskCtx.fillRect(x, y, w, h);
      }
    }

    saveCurrentMask();
  });
}

function updateBrushCursorSize() {
  const scaledSize = state.brushSize * state.zoom;
  brushCursor.style.width = `${scaledSize}px`;
  brushCursor.style.height = `${scaledSize}px`;
}

function drawBrushStroke(x, y) {
  maskCtx.globalCompositeOperation = 'source-over';
  maskCtx.fillStyle = 'rgba(239, 68, 68, 0.55)';
  maskCtx.strokeStyle = 'rgba(239, 68, 68, 0.55)';
  maskCtx.lineWidth = state.brushSize;
  maskCtx.lineCap = 'round';
  maskCtx.lineJoin = 'round';

  maskCtx.lineTo(x, y);
  maskCtx.stroke();
  maskCtx.beginPath();
  maskCtx.moveTo(x, y);
}

function eraseStroke(x, y) {
  maskCtx.globalCompositeOperation = 'destination-out';
  maskCtx.beginPath();
  maskCtx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
  maskCtx.fill();
}

function redrawWithRectPreview(currentX, currentY) {
  if (!state.maskSnapshot) return;
  maskCtx.globalCompositeOperation = 'source-over';
  maskCtx.putImageData(state.maskSnapshot, 0, 0);

  maskCtx.fillStyle = 'rgba(239, 68, 68, 0.35)';
  maskCtx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
  maskCtx.lineWidth = 2;

  const x = Math.min(state.rectStartX, currentX);
  const y = Math.min(state.rectStartY, currentY);
  const w = Math.abs(currentX - state.rectStartX);
  const h = Math.abs(currentY - state.rectStartY);
  maskCtx.fillRect(x, y, w, h);
  maskCtx.strokeRect(x, y, w, h);
}

function isMaskEmpty() {
  if (!maskCanvas.width || !maskCanvas.height) return true;
  const ctx = maskCanvas.getContext('2d');
  const imgData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const data = imgData.data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 10) return false;
  }
  return true;
}

// 5. Smart Auto-Detection & Preset Handlers
function setupAutoDetect() {
  const btnDetectTranslucentGrid = document.getElementById('btnDetectTranslucentGrid');
  const btnDetectSparkles = document.getElementById('btnDetectSparkles');
  const btnDetectLetterbox = document.getElementById('btnDetectLetterbox');

  if (btnDetectTranslucentGrid) {
    btnDetectTranslucentGrid.addEventListener('click', () => runAutoDetect('translucent_grid'));
  }
  if (btnDetectSparkles) {
    btnDetectSparkles.addEventListener('click', () => runAutoDetect('ai_sparkles'));
  }
  if (btnDetectLetterbox) {
    btnDetectLetterbox.addEventListener('click', () => runAutoDetect('letterbox'));
  }
  if (btnAutoDetect) {
    btnAutoDetect.addEventListener('click', () => runAutoDetect('all'));
  }

  // Floating Banner buttons
  if (btnProceedRemoval) {
    btnProceedRemoval.addEventListener('click', () => {
      hideDetectionBanner();
      btnProcessPage.click();
    });
  }

  if (btnDismissBoxes) {
    btnDismissBoxes.addEventListener('click', () => {
      hideDetectionBanner();
      clearMask();
    });
  }

  if (btnClearDetectedBoxes) {
    btnClearDetectedBoxes.addEventListener('click', () => {
      hideDetectionBanner();
      clearMask();
    });
  }
}

/**
 * Automatically executed on upload:
 * Detects watermarks as bounding boxes ONLY, does not modify or remove anything.
 * Prompts the user with a floating action banner to proceed or edit.
 */
async function autoDetectOnUpload() {
  if (!state.sessionId) return;
  showLoading('Auto-Detecting Watermarks...', 'Analyzing corner sparkles, badges, and overlays...');

  try {
    const res = await fetch('/api/auto-detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        page_num: state.currentPage,
        sensitivity: parseFloat(detectSensRange.value),
        dilate_pixels: parseInt(dilateRange.value),
        target_type: 'all'
      })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    handleDetectionResults(data);
  } catch (err) {
    console.warn('Auto-detect error:', err);
  } finally {
    hideLoading();
  }
}

async function runAutoDetect(targetType = 'all') {
  if (!state.sessionId) {
    alert('Please upload an image or PDF first.');
    return;
  }

  const promptText = targetType === 'ai_sparkles'
    ? 'Detecting AI Sparkles & Corner Stars...'
    : targetType === 'letterbox'
    ? 'Detecting Letterbox Badges & Overlays...'
    : targetType === 'translucent_grid'
    ? 'Detecting Translucent Grid & Stock Text...'
    : 'Detecting Watermarks...';

  showLoading(promptText, 'Scanning candidate regions');

  try {
    const res = await fetch('/api/auto-detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        page_num: state.currentPage,
        sensitivity: parseFloat(detectSensRange.value),
        dilate_pixels: parseInt(dilateRange.value),
        target_type: targetType
      })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    if (data.detected_count === 0) {
      alert('No distinct watermark detected automatically for this preset. You can freely mark it with the Brush or Box tool!');
      hideDetectionBanner();
    } else {
      handleDetectionResults(data);
    }
  } catch (err) {
    alert('Auto-detection error: ' + err.message);
  } finally {
    hideLoading();
  }
}

function handleDetectionResults(data) {
  state.detectedBoxes = data.boxes || [];
  const count = state.detectedBoxes.length;

  if (count > 0) {
    // 1. Render bounding boxes on mask canvas
    renderBoxesOnMask(state.detectedBoxes);

    // 2. Set recommended mode automatically based on watermark type
    if (data.recommended_mode) {
      removalModeSelect.value = data.recommended_mode;
    }

    // 3. Display floating banner with user option to proceed
    if (detectionBanner) {
      detectionBanner.classList.remove('hidden');
      if (detectionTitle) {
        if (data.recommended_mode === 'deblend') {
          detectionTitle.textContent = `${count} Translucent Watermark${count > 1 ? 's' : ''} Detected!`;
        } else {
          detectionTitle.textContent = `${count} Watermark${count > 1 ? 's' : ''} Detected!`;
        }
      }
      if (detectionDesc) {
        if (data.recommended_mode === 'deblend') {
          detectionDesc.textContent = `Zero-Blur Inverse Alpha activated: preserves 100% of wave textures, reflections & gradients without blur. Click "Proceed to Remove".`;
        } else {
          detectionDesc.textContent = `Watermark regions highlighted as boxes below. Click "Proceed to Remove" to clean them with AI.`;
        }
      }
    }

    // 4. Populate sidebar detected boxes list
    if (detectedItemsSection) {
      detectedItemsSection.classList.remove('hidden');
      if (detectedBoxesCountBadge) {
        detectedBoxesCountBadge.textContent = `${count} detected`;
      }
      if (detectedBoxesList) {
        detectedBoxesList.innerHTML = '';
        state.detectedBoxes.forEach((box, idx) => {
          const chip = document.createElement('div');
          chip.className = 'box-chip';
          chip.innerHTML = `
            <div class="box-chip-title">
              <span>🏷️</span>
              <span>#${idx + 1}: ${box.label || box.type}</span>
            </div>
            <button class="btn-link delete-box-btn" title="Remove this box" data-idx="${idx}">✕</button>
          `;
          chip.querySelector('.delete-box-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            removeDetectedBox(idx);
          });
          detectedBoxesList.appendChild(chip);
        });
      }
    }
  } else {
    hideDetectionBanner();
  }
}

/**
 * Draws crisp, professional bounding boxes on mask canvas:
 * - Semi-transparent fill for inpainting mask
 * - High-contrast border with corner accents
 * - Numbered floating badge label for each box
 */
function renderBoxesOnMask(boxes) {
  maskCtx.globalCompositeOperation = 'source-over';
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  const uiCtx = overlayCtx || maskCtx;
  if (overlayCtx && overlayCanvas) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  boxes.forEach((box, idx) => {
    if (box.polygon && box.polygon.length >= 3) {
      // 1. Clean filled polygon onto maskCanvas ONLY (sent to backend)
      maskCtx.beginPath();
      maskCtx.moveTo(box.polygon[0][0], box.polygon[0][1]);
      for (let i = 1; i < box.polygon.length; i++) {
        maskCtx.lineTo(box.polygon[i][0], box.polygon[i][1]);
      }
      maskCtx.closePath();
      maskCtx.fillStyle = 'rgba(239, 68, 68, 0.40)';
      maskCtx.fill();

      // 2. Crisp outline, vertex dots, and label badge onto overlayCanvas (visual only!)
      uiCtx.beginPath();
      uiCtx.moveTo(box.polygon[0][0], box.polygon[0][1]);
      for (let i = 1; i < box.polygon.length; i++) {
        uiCtx.lineTo(box.polygon[i][0], box.polygon[i][1]);
      }
      uiCtx.closePath();
      uiCtx.strokeStyle = 'rgba(239, 68, 68, 0.95)';
      uiCtx.lineWidth = 2.0;
      uiCtx.stroke();

      uiCtx.fillStyle = '#ef4444';
      box.polygon.forEach(pt => {
        uiCtx.beginPath();
        uiCtx.arc(pt[0], pt[1], 2.5, 0, Math.PI * 2);
        uiCtx.fill();
      });

      const labelText = `#${idx + 1} ${box.label || box.type || 'Watermark'}`;
      uiCtx.font = 'bold 11px Inter, system-ui, sans-serif';
      const textW = uiCtx.measureText(labelText).width;
      const lx = box.polygon[0][0];
      const ly = Math.max(16, box.polygon[0][1] - 6);

      uiCtx.fillStyle = 'rgba(15, 23, 42, 0.92)';
      uiCtx.fillRect(lx, ly - 13, textW + 10, 16);
      uiCtx.fillStyle = '#fca5a5';
      uiCtx.fillText(labelText, lx + 5, ly);
    } else {
      // 1. Clean rectangle fill onto maskCanvas ONLY (sent to backend)
      maskCtx.fillStyle = 'rgba(239, 68, 68, 0.35)';
      maskCtx.fillRect(box.x, box.y, box.width, box.height);

      // 2. Visual border, corner accents, and label badge onto overlayCanvas
      uiCtx.strokeStyle = 'rgba(239, 68, 68, 0.95)';
      uiCtx.lineWidth = 2.5;
      uiCtx.strokeRect(box.x, box.y, box.width, box.height);

      const corner = Math.min(12, Math.min(box.width, box.height) / 3);
      uiCtx.fillStyle = '#ef4444';
      uiCtx.fillRect(box.x - 2, box.y - 2, corner, 3);
      uiCtx.fillRect(box.x - 2, box.y - 2, 3, corner);
      uiCtx.fillRect(box.x + box.width - corner + 2, box.y - 2, corner, 3);
      uiCtx.fillRect(box.x + box.width - 1, box.y - 2, 3, corner);
      uiCtx.fillRect(box.x - 2, box.y + box.height - 1, corner, 3);
      uiCtx.fillRect(box.x - 2, box.y + box.height - corner + 2, 3, corner);
      uiCtx.fillRect(box.x + box.width - corner + 2, box.y + box.height - 1, corner, 3);
      uiCtx.fillRect(box.x + box.width - 1, box.y + box.height - corner + 2, 3, corner);

      const labelText = `#${idx + 1} ${box.label || box.type || 'Watermark'}`;
      uiCtx.font = 'bold 12px Inter, system-ui, sans-serif';
      const textW = uiCtx.measureText(labelText).width;
      const tagY = box.y > 24 ? box.y - 6 : box.y + 18;

      uiCtx.fillStyle = 'rgba(15, 23, 42, 0.92)';
      uiCtx.fillRect(box.x, tagY - 14, textW + 12, 18);
      uiCtx.fillStyle = '#fca5a5';
      uiCtx.fillText(labelText, box.x + 6, tagY);
    }
  });

  saveCurrentMask();
}

function removeDetectedBox(index) {
  state.detectedBoxes.splice(index, 1);
  handleDetectionResults({ boxes: state.detectedBoxes });
}

function hideDetectionBanner() {
  if (detectionBanner) detectionBanner.classList.add('hidden');
  if (detectedItemsSection) detectedItemsSection.classList.add('hidden');
}

// 6. Watermark Removal Processing
function setupProcessing() {
  btnProcessPage.addEventListener('click', async () => {
    if (!state.sessionId) {
      alert('Please upload an image or PDF first.');
      return;
    }

    if (isMaskEmpty()) {
      alert('⚠️ No watermark selected!\n\nPlease highlight the watermark using the Brush or Box tool, or click "✨ AI Sparkles" / "Auto-Detect" first.');
      return;
    }

    // Hide detection banner when processing begins
    hideDetectionBanner();

    const maskData = maskCanvas.toDataURL('image/png');
    showLoading('Removing Watermark with AI...', 'Running Big-LaMa neural inpainting for photorealistic texture synthesis');

    try {
      const payload = {
        session_id: state.sessionId,
        page_num: state.currentPage,
        mask_b64: maskData,
        mode: removalModeSelect.value,
        alpha_hint: parseFloat(opacityRange.value),
        watermark_color: colorHintSelect.value,
        inpaint_radius: 5,
        match_grain: true
      };

      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      state.cleanedImageSrc = data.cleaned_b64;

      // Update comparison stage
      imgCleaned.src = state.cleanedImageSrc;

      // Enable comparison tab and download
      btnViewComparison.disabled = false;
      btnDownload.disabled = false;
      downloadBtnText.textContent = state.fileType === 'pdf' ? 'Download Page' : 'Download Clean Image';

      // Automatically switch to comparison split view to see instant result
      switchView('comparison');
    } catch (err) {
      alert('Processing failed: ' + err.message);
    } finally {
      hideLoading();
    }
  });

  // Batch process all PDF pages
  btnProcessAllPages.addEventListener('click', async () => {
    if (!state.sessionId || state.fileType !== 'pdf') return;

    const hasPerPageMasks = Object.keys(state.pageMasks).some(k => k !== 'template');
    if (isMaskEmpty() && !hasPerPageMasks) {
      alert('⚠️ No watermark selected!\n\nPlease mark the watermark using the Box / Brush tool or click Auto-Detect before batch cleaning the PDF.');
      return;
    }

    hideDetectionBanner();
    const templateMask = isMaskEmpty() ? '' : maskCanvas.toDataURL('image/png');
    showLoading('Processing Entire PDF...', `Cleaning all ${state.pageCount} pages and reconstructing document...`);

    try {
      const pageMasksPayload = {};
      if (!state.applyAllPages) {
        Object.entries(state.pageMasks).forEach(([k, v]) => {
          if (k !== 'template' && v) pageMasksPayload[k] = v;
        });
      }

      const payload = {
        session_id: state.sessionId,
        template_mask_b64: templateMask,
        page_masks: Object.keys(pageMasksPayload).length ? pageMasksPayload : null,
        mode: removalModeSelect.value,
        alpha_hint: parseFloat(opacityRange.value),
        watermark_color: colorHintSelect.value,
        inpaint_radius: 5,
        dpi: 150
      };

      const res = await fetch('/api/process-all-pages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      btnDownload.disabled = false;
      downloadBtnText.textContent = 'Download Clean PDF';

      // Trigger download cleanly via temporary anchor so current page state stays intact
      const a = document.createElement('a');
      a.href = data.download_url;
      a.download = '';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert('Batch PDF processing failed: ' + err.message);
    } finally {
      hideLoading();
    }
  });

  // Download button
  btnDownload.addEventListener('click', () => {
    if (!state.sessionId) return;
    if (state.fileType === 'pdf' && downloadBtnText.textContent.includes('PDF')) {
      window.location.href = `/api/download/${state.sessionId}?type=pdf`;
    } else {
      window.location.href = `/api/download/${state.sessionId}?type=page&page_num=${state.currentPage}`;
    }
  });
}

// 7. Interactive Before/After Split Slider
function setupSplitSlider() {
  btnViewEditor.addEventListener('click', () => switchView('editor'));
  btnViewComparison.addEventListener('click', () => switchView('comparison'));

  const updateSplit = (clientX) => {
    const rect = splitContainer.getBoundingClientRect();
    let pos = (clientX - rect.left) / rect.width;
    pos = Math.max(0.02, Math.min(0.98, pos));

    const percent = pos * 100;
    splitOverlay.style.width = `${percent}%`;
    splitHandle.style.left = `${percent}%`;
  };

  splitHandle.addEventListener('mousedown', (e) => {
    state.isSplitDragging = true;
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.isSplitDragging) return;
    updateSplit(e.clientX);
  });

  window.addEventListener('mouseup', () => {
    state.isSplitDragging = false;
  });

  // Touch support for split slider
  splitHandle.addEventListener('touchstart', () => { state.isSplitDragging = true; });
  window.addEventListener('touchmove', (e) => {
    if (state.isSplitDragging && e.touches[0]) {
      updateSplit(e.touches[0].clientX);
    }
  });
  window.addEventListener('touchend', () => { state.isSplitDragging = false; });
}

function switchView(viewName) {
  if (viewName === 'editor') {
    btnViewEditor.classList.add('active');
    btnViewComparison.classList.remove('active');
    canvasStage.classList.remove('hidden');
    comparisonStage.classList.add('hidden');
  } else {
    btnViewEditor.classList.remove('active');
    btnViewComparison.classList.add('active');
    canvasStage.classList.add('hidden');
    comparisonStage.classList.remove('hidden');

    // Reset split slider to 50%
    splitOverlay.style.width = '50%';
    splitHandle.style.left = '50%';
  }
}

// 8. Pagination Controls (PDF)
function setupPagination() {
  btnPrevPage.addEventListener('click', () => {
    if (state.currentPage > 0) {
      state.currentPage--;
      updatePaginationUI();
      const hasMask = Boolean(state.pageMasks[state.currentPage] || (state.applyAllPages && state.pageMasks['template'] && !isMaskEmpty()));
      loadPage(state.currentPage, !hasMask);
    }
  });

  btnNextPage.addEventListener('click', () => {
    if (state.currentPage < state.pageCount - 1) {
      state.currentPage++;
      updatePaginationUI();
      const hasMask = Boolean(state.pageMasks[state.currentPage] || (state.applyAllPages && state.pageMasks['template'] && !isMaskEmpty()));
      loadPage(state.currentPage, !hasMask);
    }
  });
}

function updatePaginationUI() {
  pageIndicator.textContent = `Page ${state.currentPage + 1} of ${state.pageCount}`;
  btnPrevPage.disabled = state.currentPage === 0;
  btnNextPage.disabled = state.currentPage >= state.pageCount - 1;
}

// 9. Zoom Controls
function setupZoom() {
  btnZoomIn.addEventListener('click', () => {
    state.zoom = Math.min(3.0, state.zoom + 0.2);
    applyZoom();
  });

  btnZoomOut.addEventListener('click', () => {
    state.zoom = Math.max(0.3, state.zoom - 0.2);
    applyZoom();
  });

  btnFitScreen.addEventListener('click', fitToScreen);
}

function applyZoom() {
  zoomLevel.textContent = `${Math.round(state.zoom * 100)}%`;
  const w = Math.round(imageCanvas.width * state.zoom);
  const h = Math.round(imageCanvas.height * state.zoom);

  canvasStage.style.width = `${w}px`;
  canvasStage.style.height = `${h}px`;
  imageCanvas.style.width = `${w}px`;
  imageCanvas.style.height = `${h}px`;
  maskCanvas.style.width = `${w}px`;
  maskCanvas.style.height = `${h}px`;
  if (overlayCanvas) {
    overlayCanvas.style.width = `${w}px`;
    overlayCanvas.style.height = `${h}px`;
  }

  updateBrushCursorSize();
}

function fitToScreen() {
  if (!imageCanvas.width || !canvasViewport.clientWidth) return;
  const maxW = canvasViewport.clientWidth - 48;
  const maxH = canvasViewport.clientHeight - 48;

  const scaleW = maxW / imageCanvas.width;
  const scaleH = maxH / imageCanvas.height;
  state.zoom = Math.min(scaleW, scaleH, 1.0);
  applyZoom();
}

// Loading UI helpers
function showLoading(title, subtitle = '') {
  loadingText.textContent = title;
  loadingSubtext.textContent = subtitle;
  loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  loadingOverlay.classList.add('hidden');
}

// Run application
window.addEventListener('DOMContentLoaded', init);
