/**
 * Isolated Logo / Solid Badge / Opaque Object Removal Module (Zero-Mosaic).
 * Uses Multi-Scale SOR Harmonic Pyramid + Border-Reflected Exemplar Texture Synthesis.
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  async function remove(srcImgData, binMask, w, h, onStatus = null) {
    if (M.ai && M.ai.inpaint) {
      return await M.ai.inpaint(srcImgData, binMask, w, h, 64, onStatus);
    }
    return M.common.clientContentAwareInpaint(srcImgData, binMask, w, h, 42, true);
  }

  M.logo = { remove };
})(window);
