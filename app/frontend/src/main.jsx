import React from 'react'
import { createRoot } from 'react-dom/client'
// fonts are bundled (no CDN), so the app looks the same offline / inside Docker
import '@fontsource-variable/plus-jakarta-sans'
import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
