import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { AppProvider } from "./store";
import "./styles.css";

try {
  const t = localStorage.getItem("makeai.theme");
  if (t) document.documentElement.dataset.theme = t;
} catch { /* storage unavailable */ }

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppProvider><App /></AppProvider>
  </React.StrictMode>,
);
