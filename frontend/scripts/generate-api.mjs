import { spawnSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = path.resolve(frontendRoot, '..')
const outputPath = path.join(frontendRoot, 'src', 'generated', 'api.ts')
const schemaExporter = path.join(repoRoot, 'scripts', 'export_web_openapi.py')
const pythonPath = process.platform === 'win32'
  ? path.join(repoRoot, '.venv-web', 'Scripts', 'python.exe')
  : path.join(repoRoot, '.venv-web', 'bin', 'python')
const openapiCli = path.join(frontendRoot, 'node_modules', 'openapi-typescript', 'bin', 'cli.js')
const checkOnly = process.argv.slice(2).includes('--check')
const unexpectedArgs = process.argv.slice(2).filter((arg) => arg !== '--check')

if (unexpectedArgs.length) {
  process.stderr.write(`未知参数：${unexpectedArgs.join(' ')}\n`)
  process.exitCode = 2
}

function run(command, args, label) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    encoding: 'utf8',
    windowsHide: true,
    maxBuffer: 64 * 1024 * 1024,
  })
  if (result.error) throw new Error(`${label} 启动失败：${result.error.message}`)
  if (result.status !== 0) {
    const details = [result.stdout, result.stderr].filter(Boolean).join('\n').trim()
    throw new Error(`${label} 失败${details ? `：\n${details}` : ''}`)
  }
}

if (!unexpectedArgs.length) {
  const tempDir = mkdtempSync(path.join(tmpdir(), 'rag-arch-openapi-'))
  const schemaPath = path.join(tempDir, 'openapi.json')
  const generatedPath = path.join(tempDir, 'api.ts')
  try {
    // Avoid probing or downloading a system Python. The project venv is mandatory.
    if (!existsSync(pythonPath)) {
      throw new Error(`未找到项目虚拟环境 Python：${pythonPath}。请先在项目 .venv-web 中安装 requirements-web.txt。`)
    }
    if (!existsSync(openapiCli)) {
      throw new Error(`未找到本地 openapi-typescript：${openapiCli}。请先在 frontend 目录安装 npm 依赖。`)
    }
    run(pythonPath, [schemaExporter, '--output', schemaPath], '离线 OpenAPI 导出')
    run(process.execPath, [openapiCli, schemaPath, '--output', generatedPath,
      '--default-non-nullable', 'false'], '本地 TypeScript 类型生成')

    const generated = readFileSync(generatedPath)
    if (checkOnly) {
      let current
      try {
        current = readFileSync(outputPath)
      } catch {
        current = undefined
      }
      if (!current || !current.equals(generated)) {
        process.stderr.write('生成的 API 类型与 src/generated/api.ts 不一致，请运行 npm run generate:api。\n')
        process.exitCode = 1
      } else {
        process.stdout.write('OpenAPI 生成类型无漂移。\n')
      }
    } else {
      mkdirSync(path.dirname(outputPath), { recursive: true })
      copyFileSync(generatedPath, outputPath)
      process.stdout.write(`已更新 ${path.relative(frontendRoot, outputPath)}。\n`)
    }
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  } finally {
    rmSync(tempDir, { recursive: true, force: true })
  }
}
