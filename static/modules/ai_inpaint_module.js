/**
 * Browser-Side AI Neural Inpainting Module (MI-GAN ONNX via onnxruntime-web).
 * Caches the 28 MB quantized pipeline model in Cache Storage after first download
 * and falls back cleanly to zero-mosaic SOR harmonic inpainting if offline.
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
  const ORT_WASM_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  const MIGAN_URL = 'https://huggingface.co/anyisalin/migan-onnx/resolve/main/onnx/migan_pipeline.onnx';
  const CACHE_NAME = 'eraseon-ai-models-v1';

  let ortLoadPromise = null;
  let sessionPromise = null;
  let miganAvailable = null;

  function loadScript(src) {
    if (ortLoadPromise) return ortLoadPromise;
    ortLoadPromise = new Promise((resolve, reject) => {
      if (global.ort) {
        resolve(global.ort);
        return;
      }
      const s = document.createElement('script');
      s.src = src;
      s.async = true;
      s.onload = () => resolve(global.ort);
      s.onerror = () => reject(new Error('Failed to load onnxruntime-web'));
      document.head.appendChild(s);
    });
    return ortLoadPromise;
  }

  async function fetchModelCached(url, onStatus) {
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) {
          return await cached.arrayBuffer();
        }
        if (onStatus) {
          onStatus(
            'Removing Watermark with AI...',
            'Downloading neural inpainting model (28 MB, cached once in browser)...'
          );
        }
        const resp = await fetch(url, { mode: 'cors' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const clone = resp.clone();
        const buf = await resp.arrayBuffer();
        cache.put(url, clone).catch(() => {});
        return buf;
      } catch (_) {
        // Cache Storage unavailable or blocked; fall through to direct fetch
      }
    }
    if (onStatus) {
      onStatus(
        'Removing Watermark with AI...',
        'Downloading neural inpainting model (28 MB)...'
      );
    }
    const resp = await fetch(url, { mode: 'cors' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    return await resp.arrayBuffer();
  }

  async function ensureSession(onStatus) {
    if (miganAvailable === false) return null;
    if (sessionPromise) return sessionPromise;

    sessionPromise = (async () => {
      try {
        const ort = await loadScript(ORT_CDN);
        if (!ort) throw new Error('ort global missing');
        ort.env.wasm.wasmPaths = ORT_WASM_BASE;
        if (ort.env.wasm.numThreads === undefined || ort.env.wasm.numThreads > 1) {
          // Safe without crossOriginIsolated COOP/COEP headers on GitHub Pages
          ort.env.wasm.numThreads = 1;
        }
        const modelBuf = await fetchModelCached(MIGAN_URL, onStatus);
        if (onStatus) {
          onStatus(
            'Removing Watermark with AI...',
            'Running Neural Inpainting (MI-GAN ONNX) for photorealistic texture synthesis'
          );
        }
        const sess = await ort.InferenceSession.create(modelBuf, {
          executionProviders: ['wasm']
        });
        miganAvailable = true;
        return sess;
      } catch (err) {
        console.warn('Browser AI inpaint unavailable, using fallback:', err);
        miganAvailable = false;
        sessionPromise = null;
        return null;
      }
    })();

    return sessionPromise;
  }

  async function runMiganOnPatch(sess, imgData, binMask, w, h) {
    const ort = global.ort;
    const hw = w * h;
    const imgU8 = new Uint8Array(3 * hw);
    const mask0 = new Uint8Array(hw); // MI-GAN convention: 0 = hole, 255 = known background
    const sd = imgData.data;

    for (let i = 0; i < hw; i++) {
      const p = i * 4;
      imgU8[i] = sd[p];
      imgU8[hw + i] = sd[p + 1];
      imgU8[2 * hw + i] = sd[p + 2];
      mask0[i] = binMask[i] ? 0 : 255;
    }

    const resMap = await sess.run({
      image: new ort.Tensor('uint8', imgU8, [1, 3, h, w]),
      mask: new ort.Tensor('uint8', mask0, [1, 1, h, w])
    });
    const outName = sess.outputNames[0] || 'result';
    const rData = resMap[outName].data;

    // 2px feathered composite strictly over masked region so unmasked pixels remain bit-exact
    const dil1 = M.common.dilateBinary(binMask, w, h, 1, 1);
    const dil2 = M.common.dilateBinary(binMask, w, h, 2, 2);
    const out = new Uint8ClampedArray(sd);
    for (let i = 0; i < hw; i++) {
      const wMix = binMask[i] ? 1.0 : dil1[i] ? 0.65 : dil2[i] ? 0.25 : 0.0;
      if (wMix <= 0) continue;
      const p = i * 4;
      const rr = rData[i];
      const rg = rData[hw + i];
      const rb = rData[2 * hw + i];
      out[p] = Math.round(sd[p] * (1 - wMix) + rr * wMix);
      out[p + 1] = Math.round(sd[p + 1] * (1 - wMix) + rg * wMix);
      out[p + 2] = Math.round(sd[p + 2] * (1 - wMix) + rb * wMix);
      out[p + 3] = 255;
    }
    return new ImageData(out, w, h);
  }

  async function inpaint(srcImgData, binMask, w, h, pad = 64, onStatus = null) {
    const { dilateBinary, connectedComponents, clientContentAwareInpaint } = M.common;
    let maskPixels = 0;
    for (let i = 0; i < binMask.length; i++) {
      if (binMask[i]) maskPixels++;
    }
    if (maskPixels === 0) return srcImgData;

    const sess = await ensureSession(onStatus);
    if (!sess) {
      return clientContentAwareInpaint(srcImgData, binMask, w, h, 36, true);
    }

    if (onStatus) {
      onStatus(
        'Removing Watermark with AI...',
        'Running Neural Inpainting (MI-GAN ONNX) for photorealistic texture synthesis'
      );
    }

    const totalPixels = w * h;
    const mergedMask = dilateBinary(binMask, w, h, 14, 14);
    const { labels, comps } = connectedComponents(mergedMask, w, h);

    if (maskPixels > totalPixels * 0.40 || comps.length > 4 || Math.max(w, h) <= 768) {
      return await runMiganOnPatch(sess, srcImgData, binMask, w, h);
    }

    let working = new Uint8ClampedArray(srcImgData.data);
    for (const c of comps) {
      // Ensure crop has ample context (>= 256px when image allows) so MI-GAN sees full surroundings
      const minCtx = Math.min(256, Math.min(w, h));
      const padX = Math.max(pad, Math.ceil((minCtx - c.width) * 0.5));
      const padY = Math.max(pad, Math.ceil((minCtx - c.height) * 0.5));
      const x1 = Math.max(0, c.x - padX);
      const y1 = Math.max(0, c.y - padY);
      const x2 = Math.min(w, c.x + c.width + padX);
      const y2 = Math.min(h, c.y + c.height + padY);
      const cw = x2 - x1;
      const ch = y2 - y1;
      if (cw < 16 || ch < 16) continue;

      const patchPixels = new Uint8ClampedArray(cw * ch * 4);
      const compMask = new Uint8Array(cw * ch);
      let activeCnt = 0;
      for (let py = 0; py < ch; py++) {
        const sy = y1 + py;
        for (let px = 0; px < cw; px++) {
          const sx = x1 + px;
          const si = sy * w + sx;
          const di = py * cw + px;
          const sp = si * 4;
          const dp = di * 4;
          patchPixels[dp] = working[sp];
          patchPixels[dp + 1] = working[sp + 1];
          patchPixels[dp + 2] = working[sp + 2];
          patchPixels[dp + 3] = 255;
          if (binMask[si] && labels[si] === c.id) {
            compMask[di] = 1;
            activeCnt++;
          }
        }
      }
      if (activeCnt === 0) continue;

      const cleanedPatch = await runMiganOnPatch(sess, new ImageData(patchPixels, cw, ch), compMask, cw, ch);
      const cd = cleanedPatch.data;
      for (let py = 0; py < ch; py++) {
        const sy = y1 + py;
        for (let px = 0; px < cw; px++) {
          const sx = x1 + px;
          const si = sy * w + sx;
          const sp = si * 4;
          const dp = (py * cw + px) * 4;
          working[sp] = cd[dp];
          working[sp + 1] = cd[dp + 1];
          working[sp + 2] = cd[dp + 2];
        }
      }
    }

    return new ImageData(working, w, h);
  }

  M.ai = {
    inpaint,
    isLoaded: () => miganAvailable === true
  };
})(window);
