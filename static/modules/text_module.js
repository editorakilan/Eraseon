/**
 * Isolated Single Translucent Text / Signature / Stamp Deblending Module.
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  async function remove(srcImgData, activeMask, w, h, colorHint = 'auto', alphaHint = 0.5, onStatus = null) {
    const {
      extractGray, morphOpenGray, morphCloseGray, dilateBinary,
      harmonicMaskedFill, fastBoxBlurFloat, clientContentAwareInpaint,
      rgbToY, clamp255
    } = M.common;

    const src = srcImgData.data;
    const out = new Uint8ClampedArray(src);
    const len = w * h;
    const ySrc = extractGray(src, w, h);
    const open3 = morphOpenGray(ySrc, w, h, 3);
    const close3 = morphCloseGray(ySrc, w, h, 3);

    let isLight = true;
    let wmTargetR = 255.0, wmTargetG = 255.0, wmTargetB = 255.0;
    if (colorHint && colorHint !== 'auto' && colorHint.startsWith('#') && colorHint.length === 7) {
      wmTargetR = parseInt(colorHint.slice(1, 3), 16);
      wmTargetG = parseInt(colorHint.slice(3, 5), 16);
      wmTargetB = parseInt(colorHint.slice(5, 7), 16);
      isLight = rgbToY(wmTargetR, wmTargetG, wmTargetB) > 128.0;
    } else {
      let sumTH = 0, sumBH = 0;
      for (let i = 0; i < len; i++) {
        if (!activeMask[i]) continue;
        sumTH += Math.max(0, ySrc[i] - open3[i]);
        sumBH += Math.max(0, close3[i] - ySrc[i]);
      }
      isLight = sumTH >= sumBH;
      wmTargetR = wmTargetG = wmTargetB = isLight ? 255.0 : 0.0;
    }
    const wmTargetY = rgbToY(wmTargetR, wmTargetG, wmTargetB);

    const strokeMask = new Uint8Array(len);
    let strokeCnt = 0;
    for (let i = 0; i < len; i++) {
      if (!activeMask[i]) continue;
      const dev = isLight ? (ySrc[i] - open3[i]) : (close3[i] - ySrc[i]);
      if (dev >= 4.8) {
        strokeMask[i] = 1;
        strokeCnt++;
      }
    }
    if (strokeCnt < 8) {
      if (M.ai && M.ai.inpaint) return await M.ai.inpaint(srcImgData, activeMask, w, h, 64, onStatus);
      return clientContentAwareInpaint(srcImgData, activeMask, w, h, 32, true);
    }

    const softStroke = dilateBinary(strokeMask, w, h, 1);
    for (let i = 0; i < len; i++) {
      if (!activeMask[i]) softStroke[i] = 0;
    }

    const cleanBg = harmonicMaskedFill(src, softStroke, w, h, 24);
    const bgY = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      bgY[i] = rgbToY(cleanBg.rArr[i], cleanBg.gArr[i], cleanBg.bArr[i]);
    }
    const bgBlur2 = fastBoxBlurFloat(bgY, w, h, 2);
    const lapSq = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const d = bgY[i] - bgBlur2[i];
      lapSq[i] = d * d;
    }
    const lapVar = fastBoxBlurFloat(lapSq, w, h, 3);

    const smoothHole = new Uint8Array(len);
    let hasSmooth = false;
    for (let i = 0; i < len; i++) {
      if (!softStroke[i]) continue;
      const denom = isLight ? Math.max(30.0, wmTargetY - bgY[i]) : Math.max(30.0, bgY[i] - wmTargetY);
      const diff = isLight ? (ySrc[i] - bgY[i]) : (bgY[i] - ySrc[i]);
      const a = Math.max(-0.04, Math.min(Math.max(0.35, alphaHint * 1.15), diff / denom));
      if (Math.abs(a) < 0.025) continue;

      const p = i * 4;
      const r = src[p], g = src[p + 1], b = src[p + 2];
      const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      const cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
      const cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

      const inv = Math.max(0.45, 1.0 - a);
      const yRec = (yVal - a * wmTargetY) / inv;
      const cbRec = 128.0 + (cb - 128.0) / Math.max(0.65, 1.0 - a * 0.6);
      const crRec = 128.0 + (cr - 128.0) / Math.max(0.65, 1.0 - a * 0.6);

      out[p] = clamp255(yRec + 1.402 * (crRec - 128.0));
      out[p + 1] = clamp255(yRec - 0.344136 * (cbRec - 128.0) - 0.714136 * (crRec - 128.0));
      out[p + 2] = clamp255(yRec + 1.772 * (cbRec - 128.0));

      const hpStd = Math.sqrt(Math.max(0, lapVar[i]));
      if (a > 0.03 && hpStd < 3.2) {
        smoothHole[i] = 1;
        hasSmooth = true;
      }
    }

    if (hasSmooth) {
      if (M.ai && M.ai.inpaint) return await M.ai.inpaint(new ImageData(out, w, h), smoothHole, w, h, 64, onStatus);
      return clientContentAwareInpaint(new ImageData(out, w, h), smoothHole, w, h, 24, false);
    }
    return new ImageData(out, w, h);
  }

  M.text = { remove };
})(window);
