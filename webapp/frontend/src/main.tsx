import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './main.css'
import { Providers } from './providers'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Providers>
      <App />
    </Providers>
  </StrictMode>,
)
