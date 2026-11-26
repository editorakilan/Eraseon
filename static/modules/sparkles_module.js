/**
 * Isolated AI Corner Sparkles / Star Logo Module (Gemini / Imagen / SynthID).
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  function detectInRegion(imgData, gray, imgW, rx, ry, rw, rh, sensitivity, dilatePx) {
    const { fastBoxBlurFloat, connectedComponents } = M.common;
    const boxes = [];
    const y1 = ry + Math.floor(rh * 0.70);
    let y2 = ry + rh;
    while (y2 > y1 + 32) {
      let rowSum = 0;
      const off = (y2 - 1) * imgW + rx;
      for (let x = 0; x < rw; x++) rowSum += gray[off + x];
      if (rowSum / rw < 48) y2--;
      else break;
    }
    const x1 = rx + Math.floor(rw * 0.72);
    const x2 = rx + rw;
    const cw = x2 - x1;
    const ch = y2 - y1;
    if (cw < 30 || ch < 30) return boxes;

    const subGray = new Float32Array(cw * ch);
    const sampleVals = [];
    for (let yy = 0; yy < ch; yy++) {
      for (let xx = 0; xx < cw; xx++) {
        const v = gray[(y1 + yy) * imgW + (x1 + xx)];
        subGray[yy * cw + xx] = v;
        if ((yy + xx) % 4 === 0) sampleVals.push(v);
      }
    }
    sampleVals.sort((a, b) => a - b);
    if ((sampleVals[Math.floor(sampleVals.length / 2)] || 0) > 235) return boxes;

    const scaleR = Math.max(12, Math.round(Math.min(rw, rh) * 0.045));
    const blur = fastBoxBlurFloat(subGray, cw, ch, scaleR);
    const thThresh = Math.max(12, 21 - sensitivity * 11);
    const bin = new Uint8Array(cw * ch);
    for (let i = 0; i < bin.length; i++) {
      if (subGray[i] - blur[i] >= thThresh) bin[i] = 1;
    }

    const minSize = Math.max(10, Math.round(Math.min(rw, rh) * 0.025));
    const maxSize = Math.max(64, Math.round(Math.min(rw, rh) * 0.14));
    const { comps } = connectedComponents(bin, cw, ch);
    const pad = Math.max(5, dilatePx);
    for (const c of comps) {
      const aspect = c.width / Math.max(1, c.height);
      if (c.width >= minSize && c.width <= maxSize && c.height >= minSize && c.height <= maxSize && aspect >= 0.70 && aspect <= 1.42 && c.area >= 35) {
        let colorDiff = 0;
        let maxBright = 0;
        let cnt = 0;
        for (let yy = c.y; yy < c.y + c.height; yy++) {
          for (let xx = c.x; xx < c.x + c.width; xx++) {
            const p = ((y1 + yy) * imgW + (x1 + xx)) * 4;
            colorDiff += Math.abs(imgData[p] - imgData[p + 1]) + Math.abs(imgData[p + 1] - imgData[p + 2]);
            const v = subGray[yy * cw + xx];
            if (v > maxBright) maxBright = v;
            cnt++;
          }
        }
        if (cnt > 0 && colorDiff / cnt < 55 && maxBright >= 180) {
          boxes.push({
            x: Math.max(0, x1 + c.x - pad),
            y: Math.max(0, y1 + c.y - pad),
            width: c.width + pad * 2,
            height: c.height + pad * 2,
            type: 'ai_sparkles',
            label: 'AI Corner Sparkle'
          });
        }
      }
    }
    return boxes;
  }

  async function remove(srcImgData, binMask, w, h, onStatus = null) {
    const procMask = M.common.dilateBinary(binMask, w, h, 7, 7);
    return M.common.clientContentAwareInpaint(srcImgData, procMask, w, h, 44, true);
  }

  M.sparkles = { detectInRegion, remove };
})(window);
