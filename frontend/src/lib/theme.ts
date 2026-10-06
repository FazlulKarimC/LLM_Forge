export const THEME_KEY = "llmforge.theme.light";
// Runs before page paint; storage denial falls back to the default dark theme.
export const themeBootstrap = `try{document.documentElement.dataset.theme=localStorage.getItem("${THEME_KEY}")==="true"?"light":"dark"}catch{document.documentElement.dataset.theme="dark"}`;
