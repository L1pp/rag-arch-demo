export function isPageViewerPath(pathname: string) {
  return pathname.replace(/\/+$/, '') === '/viewer/pages'
}

export function isLoginPath(pathname: string) {
  return pathname.replace(/\/+$/, '') === '/login'
}

/** Return only a same-origin absolute path; login redirects never accept a URL. */
export function safeNextPath(value: string | null | undefined, origin = window.location.origin) {
  if (!value || !value.startsWith('/') || value.startsWith('//') || /[\\\u0000-\u001f\u007f]/.test(value)) return '/'
  try {
    const target = new URL(value, origin)
    if (target.origin !== origin || target.username || target.password) return '/'
    if (isLoginPath(target.pathname)) return '/'
    return `${target.pathname}${target.search}${target.hash}`
  } catch {
    return '/'
  }
}
