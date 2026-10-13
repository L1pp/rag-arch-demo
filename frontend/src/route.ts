export function isPageViewerPath(pathname: string) {
  return pathname.replace(/\/+$/, '') === '/viewer/pages'
}
