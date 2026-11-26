/**
 * Isolated Classical Inpainting Module (Telea FMM / Navier-Stokes Smooth Gradient).
 */
(function (global) {
  const M = (global.EraseonModules = global.EraseonModules || {});

  function removeTelea(srcImgData, binMask, w, h) {
    return M.common.clientContentAwareInpaint(srcImgData, binMask, w, h, 32, false);
  }

  function removeNS(srcImgData, binMask, w, h) {
    return M.common.clientContentAwareInpaint(srcImgData, binMask, w, h, 48, false);
  }

  M.inpaint = { removeTelea, removeNS };
})(window);
