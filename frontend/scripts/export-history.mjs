import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url))
const repositoryRoot = path.resolve(scriptDirectory, '..', '..')
const pythonScript = path.join(repositoryRoot, 'scripts', 'export_history_viewer.py')
const python = process.platform === 'win32'
  ? path.join(repositoryRoot, '.venv-web', 'Scripts', 'python.exe')
  : path.join(repositoryRoot, '.venv-web', 'bin', 'python')
if (!existsSync(python)) {
  console.error('缺少项目虚拟环境 Python（.venv-web），请准备项目环境后重试。')
  process.exit(1)
}
const forwardedArguments = process.argv.slice(2)
const outputArgument = forwardedArguments.indexOf('--output')
if (outputArgument >= 0 && outputArgument + 1 < forwardedArguments.length) {
  const requestedOutput = forwardedArguments[outputArgument + 1]
  if (!path.isAbsolute(requestedOutput)
      && requestedOutput.replaceAll('\\', '/') === 'public/viewer/history') {
    forwardedArguments[outputArgument + 1] = 'frontend/public/viewer/history'
  }
}

const result = spawnSync(python, [pythonScript, ...forwardedArguments], {
  cwd: repositoryRoot,
  stdio: 'inherit',
  windowsHide: true,
})

if (result.error) {
  console.error('无法启动项目虚拟环境中的 Python 历史导出器。')
  process.exitCode = 1
} else {
  process.exitCode = result.status ?? 1
}
