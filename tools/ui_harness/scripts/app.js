/* Stand-in for ComfyUI's scripts/app.js, just enough for the editor UI. */
export const app = {
  __extensions: [],
  registerExtension(extension) {
    app.__extensions.push(extension);
  },
  graph: {
    _nodes: [],
    setDirtyCanvas() {},
    getNodeById() {
      return null;
    },
  },
  async refreshComboInNodes() {},
  ui: {},
};
