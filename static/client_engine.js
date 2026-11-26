/**
 * Eraseon - Pure Browser Client-Side Engine Orchestrator
 * Routes each operation strictly to its isolated module inside window.EraseonModules.
 */
(function (global) {
  const PDFJS_CDN = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js';
  const PDFJS_WORKER_CDN = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';
  const PDFLIB_CDN = 'https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js';

  const loadedScripts = {};
  function loadScriptOnce(src) {
    if (loadedScripts[src]) return loadedScripts[src];
    loadedScripts[src] = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
    return loadedScripts[src];
  }

  async function ensurePdfJs() {
    await loadScriptOnce(PDFJS_CDN);
    if (global.pdfjsLib && !global.pdfjsLib.GlobalWorkerOptions.workerSrc) {
      global.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_CDN;
    }
    return global.pdfjsLib;
  }

  async function ensurePdfLib() {
    await loadScriptOnce(PDFLIB_CDN);
    return global.PDFLib;
  }

  const clientSessions = {};

  function autoDetectClient(imageData, sensitivity = 0.5, targetType = 'all', dilatePx = 4) {
    const M = global.EraseonModules;
    const { extractGray, findEmbeddedPhotoRegions } = M.common;
    const { data, width: w, height: h } = imageData;
    const gray = extractGray(data, w, h);
    const embedded = findEmbeddedPhotoRegions(gray, w, h);
    const scanRegions = embedded.length > 0 ? embedded : [{ x: 0, y: 0, width: w, height: h }];

    const boxes = [];
    if (targetType === 'all' || targetType === 'ai_sparkles' || targetType === 'sparkles') {
      for (const r of scanRegions) {
        boxes.push(...M.sparkles.detectInRegion(data, gray, w, r.x, r.y, r.width, r.height, sensitivity, dilatePx));
      }
    }
    if (targetType === 'all' || targetType === 'letterbox') {
      for (const r of scanRegions) {
        boxes.push(...M.letterbox.detectLetterboxInRegion(gray, w, r.x, r.y, r.width, r.height));
        boxes.push(...M.letterbox.detectContainerBadgesInRegion(gray, w, r.x, r.y, r.width, r.height));
      }
    }
    let gridCount = 0;
    if (targetType === 'all' || targetType === 'text' || targetType === 'translucent_grid' || targetType === 'grid') {
      const allowNonRep = targetType === 'translucent_grid' || targetType === 'grid';
      for (const r of scanRegions) {
        const gridBoxes = M.grid.detectInRegion(gray, w, h, r.x, r.y, r.width, r.height, sensitivity, dilatePx, allowNonRep);
        gridCount += gridBoxes.length;
        boxes.push(...gridBoxes);
      }
    }

    const diagCount = boxes.filter((b) => b.angle && b.angle > 8).length;
    const hasLetterbox = boxes.some((b) => b.type === 'letterbox_bar' || b.type === 'container_badge');
    const hasSparkles = boxes.some((b) => b.type === 'ai_sparkles');
    let recommendedMode = 'auto';
    if (gridCount >= 3 || diagCount >= 2) recommendedMode = 'grid';
    else if (hasLetterbox && !hasSparkles && gridCount === 0) recommendedMode = 'letterbox';
    else if (hasSparkles && !hasLetterbox && gridCount === 0) recommendedMode = 'sparkles';

    return {
      detected_count: boxes.length,
      boxes,
      recommended_mode: recommendedMode
    };
  }

  async function processImageDataClient(srcImgData, maskImgData, mode = 'auto', colorHint = 'auto', alphaHint = 0.5, boxesHint = null, onStatus = null) {
    const M = global.EraseonModules;
    const { enhanceCleanedImage } = M.common;
    const w = srcImgData.width;
    const h = srcImgData.height;
    const binMask = new Uint8Array(w * h);
    const mData = maskImgData.data;
    let maskPixels = 0;
    for (let i = 0, p = 3; i < binMask.length; i++, p += 4) {
      if (mData[p] > 10) {
        binMask[i] = 1;
        maskPixels++;
      }
    }
    if (maskPixels === 0) return srcImgData;

    const m = (mode || 'auto').toLowerCase().trim();

    // Explicit isolated module dispatch (zero cross-module interference)
    if (m === 'logo' || m === 'lama') {
      return enhanceCleanedImage(await M.logo.remove(srcImgData, binMask, w, h, onStatus));
    }
    if (m === 'grid' || m === 'deblend') {
      return enhanceCleanedImage(await M.grid.remove(srcImgData, binMask, w, h, colorHint, alphaHint, boxesHint, onStatus));
    }
    if (m === 'sparkles' || m === 'ai_sparkles') {
      return enhanceCleanedImage(await M.sparkles.remove(srcImgData, binMask, w, h, onStatus));
    }
    if (m === 'letterbox') {
      return enhanceCleanedImage(await M.letterbox.remove(srcImgData, binMask, w, h, onStatus));
    }
    if (m === 'text') {
      return enhanceCleanedImage(await M.text.remove(srcImgData, binMask, w, h, colorHint, alphaHint, onStatus));
    }
    if (m === 'inpaint_telea') {
      return enhanceCleanedImage(M.inpaint.removeTelea(srcImgData, binMask, w, h));
    }
    if (m === 'inpaint_ns') {
      return enhanceCleanedImage(M.inpaint.removeNS(srcImgData, binMask, w, h));
    }

    // Smart Auto-Router: route each detected region type to its dedicated module
    let workingImg = srcImgData;
    let remainingBinMask = binMask;
    let remainingBoxes = boxesHint;

    if (Array.isArray(boxesHint) && boxesHint.length > 0) {
      const lbBoxes = boxesHint.filter((b) => b.type === 'letterbox_bar' || b.type === 'container_badge');
      const spBoxes = boxesHint.filter((b) => b.type === 'ai_sparkles');
      const transBoxes = boxesHint.filter((b) => b.type === 'translucent_grid' || b.type === 'watermark_text');

      const carveBoxMask = (boxList) => {
        const subMask = new Uint8Array(w * h);
        let cnt = 0;
        for (const sb of boxList) {
          const x0 = Math.max(0, Math.floor(sb.x) - 4), y0 = Math.max(0, Math.floor(sb.y) - 4);
          const x1 = Math.min(w, Math.ceil(sb.x + sb.width) + 4), y1 = Math.min(h, Math.ceil(sb.y + sb.height) + 4);
          for (let yy = y0; yy < y1; yy++) {
            const off = yy * w;
            for (let xx = x0; xx < x1; xx++) {
              if (remainingBinMask[off + xx]) { subMask[off + xx] = 1; cnt++; }
            }
          }
        }
        return { subMask, cnt };
      };

      const subtractMask = (subMask) => {
        const nextMask = new Uint8Array(w * h);
        for (let i = 0; i < nextMask.length; i++) {
          if (remainingBinMask[i] && !subMask[i]) nextMask[i] = 1;
        }
        remainingBinMask = nextMask;
      };

      if (lbBoxes.length > 0) {
        const { subMask, cnt } = carveBoxMask(lbBoxes);
        if (cnt > 0) {
          workingImg = await M.letterbox.remove(workingImg, subMask, w, h, onStatus);
          subtractMask(subMask);
        }
      }
      if (spBoxes.length > 0) {
        const { subMask, cnt } = carveBoxMask(spBoxes);
        if (cnt > 0) {
          workingImg = await M.sparkles.remove(workingImg, subMask, w, h, onStatus);
          subtractMask(subMask);
        }
      }
      remainingBoxes = transBoxes;
    }

    let remCount = 0;
    for (let i = 0; i < remainingBinMask.length; i++) {
      if (remainingBinMask[i]) remCount++;
    }
    if (remCount === 0) return enhanceCleanedImage(workingImg);

    let cleaned;
    if (Array.isArray(remainingBoxes) && remainingBoxes.filter((b) => b.type === 'translucent_grid').length >= 3) {
      cleaned = await M.grid.remove(workingImg, remainingBinMask, w, h, colorHint, alphaHint, remainingBoxes, onStatus);
    } else {
      cleaned = await M.logo.remove(workingImg, remainingBinMask, w, h, onStatus);
    }
    return enhanceCleanedImage(cleaned);
  }

  async function uploadClientFile(file) {
    const sessionId = 'local-' + Math.random().toString(36).slice(2, 10);
    const isPdf = file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';

    if (isPdf) {
      const pdfjsLib = await ensurePdfJs();
      const buf = await file.arrayBuffer();
      const pdfDoc = await pdfjsLib.getDocument({ data: buf }).promise;
      clientSessions[sessionId] = {
        fileType: 'pdf',
        filename: file.name,
        pageCount: pdfDoc.numPages,
        pdfDoc,
        pageCache: {}
      };
      return {
        session_id: sessionId,
        file_type: 'pdf',
        filename: file.name,
        page_count: pdfDoc.numPages
      };
    } else {
      const objectUrl = URL.createObjectURL(file);
      clientSessions[sessionId] = {
        fileType: 'image',
        filename: file.name,
        pageCount: 1,
        objectUrl
      };
      return {
        session_id: sessionId,
        file_type: 'image',
        filename: file.name,
        page_count: 1
      };
    }
  }

  async function getClientPageUrl(sessionId, pageNum) {
    const sess = clientSessions[sessionId];
    if (!sess) throw new Error('Session not found');
    if (sess.fileType === 'image') return sess.objectUrl;
    if (sess.pageCache[pageNum]) return sess.pageCache[pageNum];

    const page = await sess.pdfDoc.getPage(pageNum + 1);
    const viewport = page.getViewport({ scale: 2.0 });
    const c = document.createElement('canvas');
    c.width = Math.round(viewport.width);
    c.height = Math.round(viewport.height);
    const ctx = c.getContext('2d');
    await page.render({ canvasContext: ctx, viewport }).promise;
    const url = c.toDataURL('image/png');
    sess.pageCache[pageNum] = url;
    return url;
  }

  async function processAllPdfClient(sessionId, templateMaskDataUrl, pageMasksMap, mode, colorHint, alphaHint = 0.5) {
    const sess = clientSessions[sessionId];
    if (!sess || sess.fileType !== 'pdf') throw new Error('PDF session not found');
    const PDFLib = await ensurePdfLib();
    const outPdf = await PDFLib.PDFDocument.create();

    const loadImg = (url) => new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = url;
    });

    for (let p = 0; p < sess.pageCount; p++) {
      const pageUrl = await getClientPageUrl(sessionId, p);
      const pageImg = await loadImg(pageUrl);
      const pw = pageImg.naturalWidth;
      const ph = pageImg.naturalHeight;

      const c = document.createElement('canvas');
      c.width = pw;
      c.height = ph;
      const ctx = c.getContext('2d');
      ctx.drawImage(pageImg, 0, 0);

      const maskUrl = (pageMasksMap && pageMasksMap[p]) ? pageMasksMap[p] : templateMaskDataUrl;
      if (maskUrl) {
        const mImg = await loadImg(maskUrl);
        const mc = document.createElement('canvas');
        mc.width = pw;
        mc.height = ph;
        const mctx = mc.getContext('2d');
        mctx.drawImage(mImg, 0, 0, pw, ph);

        const srcData = ctx.getImageData(0, 0, pw, ph);
        const maskData = mctx.getImageData(0, 0, pw, ph);
        const cleaned = await processImageDataClient(srcData, maskData, mode, colorHint, alphaHint);
        ctx.putImageData(cleaned, 0, 0);
      }

      const pngBytes = await new Promise((resolve) => c.toBlob((b) => b.arrayBuffer().then(resolve), 'image/png'));
      const embedded = await outPdf.embedPng(pngBytes);
      const ptW = pw * 0.5;
      const ptH = ph * 0.5;
      const pdfPage = outPdf.addPage([ptW, ptH]);
      pdfPage.drawImage(embedded, { x: 0, y: 0, width: ptW, height: ptH });
    }

    const pdfBytes = await outPdf.save();
    return new Blob([pdfBytes], { type: 'application/pdf' });
  }

  global.EraseonClientEngine = {
    uploadClientFile,
    getClientPageUrl,
    autoDetectClient,
    processImageDataClient,
    processAllPdfClient
  };
})(window);
