import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import PageViewer from './PageViewer'
import { isPageViewerPath } from './route'
import { WorkspaceProvider } from './WorkspaceContext'
import './styles.css'
import './layout.css'

export function RootApp() {
  return isPageViewerPath(window.location.pathname)
    ? <PageViewer />
    : <WorkspaceProvider><App /></WorkspaceProvider>
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootApp />
  </React.StrictMode>,
)
