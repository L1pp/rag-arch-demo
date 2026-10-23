import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import AuthBoundary from './AuthBoundary'
import LoginPage from './LoginPage'
import PageViewer from './PageViewer'
import { isLoginPath, isPageViewerPath } from './route'
import { WorkspaceProvider } from './WorkspaceContext'
import './styles.css'
import './layout.css'
import './auth.css'

export function RootApp() {
  const pathname = window.location.pathname
  if (isLoginPath(pathname)) return <LoginPage />
  return <AuthBoundary>{isPageViewerPath(pathname)
    ? <PageViewer />
    : <WorkspaceProvider><App /></WorkspaceProvider>}</AuthBoundary>
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootApp />
  </React.StrictMode>,
)
