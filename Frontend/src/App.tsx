import { useEffect, useRef, useState, type DragEvent } from 'react'
import './App.css'

// SVG Icon Components
const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
)

const FileIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
)

const AnalysisIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
    <path d="M11 8v6" />
    <path d="M8 11h6" />
  </svg>
)

const CoordinatesIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <circle cx="15.5" cy="15.5" r="1.5" />
    <circle cx="8.5" cy="15.5" r="1.5" />
    <circle cx="15.5" cy="8.5" r="1.5" />
  </svg>
)

const DownloadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
)

const LoadingSpinner = () => (
  <svg className="spinner" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
)

const CheckIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
)

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
)

const ComponentIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2L2 7l10 5 10-5-10-5z" />
    <path d="M2 17l10 5 10-5" />
    <path d="M2 12l10 5 10-5" />
  </svg>
)

type FrameSummary = {
  page_index: number
  width: number
  height: number
  mode: string
  preprocessed_width: number
  preprocessed_height: number
  preview_png_base64: string
}

type Stage1Response = {
  filename: string
  content_type: string | null
  source_type: 'image' | 'pdf'
  page_count: number
  frames: FrameSummary[]
}

type CategoryCounts = {
  motor: number
  pump: number
  tank: number
  valve: number
}

type ModelDetectionResult = {
  page_index: number
  model: string
  counts: CategoryCounts
}

type PageDetectionResult = {
  page_index: number
  counts: CategoryCounts
  model_results: ModelDetectionResult[]
}

type DetectionResponse = {
  filename: string
  content_type: string | null
  source_type: 'image' | 'pdf'
  page_count: number
  models_used: string[]
  pages: PageDetectionResult[]
  industry: string | null
}

type BatchAnalysisItem = {
  filename: string
  content_type: string | null
  source_type: 'image' | 'pdf'
  page_count: number
  result: Stage1Response
  detection: DetectionResponse
  coordinates: CoordinateDetectionResponse
  error: string | null
}

type BatchAnalysisResponse = {
  files: BatchAnalysisItem[]
}

type ComponentPosition = {
  x: number
  y: number
  width: number
  height: number
}

type ComponentMeta = {
  name: string
}

type ComponentChild = {
  meta: ComponentMeta
  position: ComponentPosition
  type: string
}

type RootMeta = {
  name: string
}

type Root = {
  children: ComponentChild[]
  meta: RootMeta
  type: string
}

type CoordinateDetectionResponse = {
  custom: Record<string, unknown>
  params: Record<string, unknown>
  props: Record<string, unknown>
  root: Root
}

type BatchRun = {
  file: File
  result: Stage1Response | null
  detection: DetectionResponse | null
  coordinates: CoordinateDetectionResponse | null
  error: string | null
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Stage1Response | null>(null)
  const [detection, setDetection] = useState<DetectionResponse | null>(null)
  const [coordinates, setCoordinates] = useState<CoordinateDetectionResponse | null>(null)
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState<string | null>(null)
  const [batchRuns, setBatchRuns] = useState<BatchRun[]>([])
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'coordinates'>('upload')

  const currentFile = selectedFiles[0] ?? null

  const totalDetectedComponents = detection
    ? detection.pages.reduce((sum, page) => {
        const c = page.counts
        return sum + (c.motor ?? 0) + (c.pump ?? 0) + (c.tank ?? 0) + (c.valve ?? 0)
      }, 0)
    : 0

  const batchTotalDetectedComponents = batchRuns.reduce((sum, run) => {
    const runTotal = run.detection
      ? run.detection.pages.reduce((pageSum, page) => {
          const c = page.counts
          return pageSum + (c.motor ?? 0) + (c.pump ?? 0) + (c.tank ?? 0) + (c.valve ?? 0)
        }, 0)
      : 0
    return sum + runTotal
  }, 0)

  const displayedDetectedComponents = batchRuns.length > 0 ? batchTotalDetectedComponents : totalDetectedComponents
  const displayedCoordinateCount = coordinates ? coordinates.root.children.length : 0

  const supportsPreview = currentFile
    ? currentFile.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(currentFile.name)
    : false

  const isTextCoordinate = (component: ComponentChild) => component.type === 'ia.symbol.text' || component.type.endsWith('.text')

  const sanitizeCoordinates = (payload: CoordinateDetectionResponse | null) => {
    if (!payload) {
      return payload
    }

    return {
      ...payload,
      root: {
        ...payload.root,
        children: payload.root.children.filter((component) => !isTextCoordinate(component)),
      },
    }
  }

  const handleFilesSelect = (files: File[]) => {
    setSelectedFiles(files)
    setError(null)
    setResult(null)
    setDetection(null)
    setCoordinates(null)
    setBatchRuns([])
    setActiveTab('upload')
  }

  const postFile = async (endpoint: string, file: File) => {
    if (!file) {
      throw new Error('Choose an image or PDF first.')
    }

    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null
      throw new Error(payload?.detail ?? `Request failed with status ${response.status}`)
    }

    return response.json()
  }

  const postFiles = async (endpoint: string, files: File[]) => {
    if (!files.length) {
      throw new Error('Choose one or more images or PDFs first.')
    }

    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null
      throw new Error(payload?.detail ?? `Request failed with status ${response.status}`)
    }

    return response.json()
  }

  // Execute upload, detection and coordinate detection sequentially.
  const runAll = async () => {
    if (!selectedFiles.length) {
      setError('Choose one or more images or PDFs first.')
      return
    }

    setIsUploading(true)
    setError(null)
    try {
      const runs: BatchRun[] = []

      if (selectedFiles.length === 1) {
        const file = selectedFiles[0]
        try {
          const uploadPayload = (await postFile('/upload', file)) as Stage1Response
          const detectionPayload = (await postFile('/detect', file)) as DetectionResponse
          const coordinatesPayload = sanitizeCoordinates((await postFile('/coordinates', file)) as CoordinateDetectionResponse)
          runs.push({
            file,
            result: uploadPayload,
            detection: detectionPayload,
            coordinates: coordinatesPayload,
            error: null,
          })
        } catch (fileError) {
          runs.push({
            file,
            result: null,
            detection: null,
            coordinates: null,
            error: fileError instanceof Error ? fileError.message : 'Run failed for file.',
          })
        }
      } else {
        const batchResponse = (await postFiles('/analyze_batch', selectedFiles)) as BatchAnalysisResponse

        batchResponse.files.forEach((item, index) => {
          const file = selectedFiles[index]
          runs.push({
            file,
            result: item.result ?? null,
            detection: item.detection ?? null,
            coordinates: sanitizeCoordinates(item.coordinates ?? null),
            error: item.error ?? null,
          })
        })
      }

      setBatchRuns(runs)
      const firstSuccess = runs.find((run) => run.result && run.detection && run.coordinates) ?? null
      setResult(firstSuccess?.result ?? null)
      setDetection(firstSuccess?.detection ?? null)
      setCoordinates(firstSuccess?.coordinates ?? null)
      setActiveTab('coordinates')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Run All failed.')
    } finally {
      setIsUploading(false)
    }
  }

  // State for the downloadable JSON URL
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)

  // Generate download URL whenever we have all data
  useEffect(() => {
    if (batchRuns.length) {
      const combined = JSON.stringify({ runs: batchRuns }, null, 2)
      const blob = new Blob([combined], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      setDownloadUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setDownloadUrl(null)
    }
  }, [batchRuns])

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    handleFilesSelect(Array.from(event.dataTransfer.files))
  }

  useEffect(() => {
    if (!currentFile || !supportsPreview) {
      setSelectedPreviewUrl(null)
      return
    }

    const previewUrl = URL.createObjectURL(currentFile)
    setSelectedPreviewUrl(previewUrl)

    return () => URL.revokeObjectURL(previewUrl)
  }, [currentFile, supportsPreview])

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <h2>P&ID Dashboard</h2>
          <p className="brand-tagline">AI-Powered Analysis</p>
        </div>

        <nav className="nav">
          <button
            className="primary-button"
            onClick={runAll}
            disabled={isUploading || selectedFiles.length === 0}
          >
            {isUploading ? (
              <span className="button-content">
                <LoadingSpinner />
                Processing...
              </span>
            ) : (
              <span className="button-content">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Run Analysis
              </span>
            )}
          </button>
          {downloadUrl && (
            <a href={downloadUrl} download="output.json" className="secondary-button download-button">
              <span className="button-content">
                <DownloadIcon />
                Download JSON
              </span>
            </a>
          )}
          <button 
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <span className="nav-item-content">
              <UploadIcon />
              Upload
            </span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
            disabled={!detection}
          >
            <span className="nav-item-content">
              <AnalysisIcon />
              Analysis
            </span>
          </button>
          <button 
            className={`nav-item ${activeTab === 'coordinates' ? 'active' : ''}`}
            onClick={() => setActiveTab('coordinates')}
            disabled={!coordinates}
          >
            <span className="nav-item-content">
              <CoordinatesIcon />
              Coordinates
            </span>
          </button>
        </nav>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <div className="eyebrow">
              {detection?.industry ? `Industry: ${detection.industry}` : 'Stage 1 - Image Input'}
            </div>
            <h1>P&ID Analysis Dashboard</h1>
            <p className="subtitle">Upload P&ID diagrams for AI-powered component detection and analysis</p>
          </div>
          <div className="topbar-right">
            <div className="status-indicator">
              <span className="status-dot"></span>
              <span>System Ready</span>
            </div>
            <div className="meta-row">
              <span>Backend: {API_BASE_URL}</span>
              <span>Accepts PNG, JPG, WEBP, BMP, TIFF, PDF</span>
            </div>
          </div>
        </header>

        <main className="shell">
          {activeTab === 'upload' && (
            <>
              <section className="hero-panel compact-hero">
                <p className="lede">
                  Drop a P&ID image or PDF, send it to the backend, and inspect the loaded
                  frames before detection starts.
                </p>
              </section>

              <section className="workspace">
                <div
                  className={`upload-card ${isDragging ? 'dragging' : ''}`}
                  onDragEnter={(event) => {
                    event.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={(event) => {
                    event.preventDefault()
                    setIsDragging(false)
                  }}
                  onDrop={handleDrop}
                >
                  <div className="upload-copy">
                    <div className="pill">Upload</div>
                    <h2>Choose the source file</h2>
                    <p>
                      The backend will render PDF pages, preprocess the image, and return a
                      per-page preview payload.
                    </p>
                    <div className="supported-formats">
                      <span className="format-tag">PNG</span>
                      <span className="format-tag">JPG</span>
                      <span className="format-tag">WEBP</span>
                      <span className="format-tag">PDF</span>
                    </div>
                  </div>

                  <div className="upload-controls">
                    <input
                      ref={fileInputRef}
                      className="file-input"
                      type="file"
                      multiple
                      accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf"
                      onChange={(event) => handleFilesSelect(Array.from(event.target.files ?? []))}
                    />
                    <button type="button" className="secondary-button browse-button" onClick={() => fileInputRef.current?.click()}>
                      <span className="button-content">
                        <FileIcon />
                        Browse files
                      </span>
                    </button>
                    <p className="upload-hint">or drag and drop your file here</p>
                  </div>

                  <div className="selection-row">
                    <div>
                      <span className="label">Selected file</span>
                      <strong>{currentFile?.name ?? 'None yet'}</strong>
                    </div>
                    <div>
                      <span className="label">Type</span>
                      <strong>{currentFile?.type || 'Unknown'}</strong>
                    </div>
                    <div>
                      <span className="label">Size</span>
                      <strong>{currentFile ? `${(currentFile.size / 1024).toFixed(1)} KB` : '--'}</strong>
                    </div>
                  </div>

                  <div className="selection-row">
                    <div>
                      <span className="label">Queued files</span>
                      <strong>{selectedFiles.length}</strong>
                    </div>
                    <div>
                      <span className="label">Batch mode</span>
                      <strong>{selectedFiles.length > 1 ? 'Enabled' : 'Single file'}</strong>
                    </div>
                  </div>

                  {error ? (
                    <div className="status error">
                      <span className="status-icon"><AlertIcon /></span>
                      {error}
                    </div>
                  ) : null}
                  {!error && result && !detection ? (
                    <div className="status success">
                      <span className="status-icon"><CheckIcon /></span>
                      Stage 1 completed successfully. Ready for AI analysis.
                    </div>
                  ) : null}
                  {!error && detection ? (
                    <div className="status success">
                      <span className="status-icon"><CheckIcon /></span>
                      AI detection completed! Industry identified: {detection.industry || 'Unknown'}
                    </div>
                  ) : null}
                  {!error && coordinates ? (
                    <div className="status success">
                      <span className="status-icon"><CheckIcon /></span>
                      Coordinate detection completed! {displayedCoordinateCount} visible components found.
                    </div>
                  ) : null}

                  {batchRuns.length > 1 ? (
                    <section className="results-grid compact">
                      <article className="result-card wide">
                        <div className="card-header">
                          <div>
                            <div className="pill muted">Batch</div>
                            <h2>Processed files</h2>
                          </div>
                        </div>
                        <div className="frame-grid">
                          {batchRuns.map((run) => (
                            <div className="frame-card" key={run.file.name}>
                              <div className="frame-meta">
                                <strong>{run.file.name}</strong>
                                <span>{run.error ? `Error: ${run.error}` : 'Completed'}</span>
                                <span>Industry: {run.detection?.industry || 'Unknown'}</span>
                                <span>
                                  {run.detection
                                    ? run.detection.pages.reduce((sum, page) => {
                                        const c = page.counts
                                        return sum + (c.motor ?? 0) + (c.pump ?? 0) + (c.tank ?? 0) + (c.valve ?? 0)
                                      }, 0)
                                    : 0}{' '}
                                  components
                                </span>
                                <span>
                                  Coordinates: {run.coordinates?.root.children.length ?? 0}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </article>
                    </section>
                  ) : null}

                  {result ? (
                    <section className="results-grid compact">
                      <article className="result-card">
                        <div className="card-header">
                          <div>
                            <div className="pill muted">Response</div>
                            <h2>Stage 1 output</h2>
                          </div>
                        </div>

                        <div className="response-summary">
                          <div><span>Filename</span><strong>{result.filename}</strong></div>
                          <div><span>Source</span><strong>{result.source_type}</strong></div>
                          <div><span>Pages</span><strong>{result.page_count}</strong></div>
                          <div><span>Content type</span><strong>{result.content_type ?? 'n/a'}</strong></div>
                        </div>
                      </article>

                      <article className="result-card wide">
                        <div className="card-header">
                          <div>
                            <div className="pill muted">Frames</div>
                            <h2>Preprocessed pages</h2>
                          </div>
                        </div>

                        <div className="frame-grid">
                          {result.frames.length ? (
                            result.frames.map((frame) => (
                              <div className="frame-card" key={frame.page_index}>
                                <img
                                  src={`data:image/png;base64,${frame.preview_png_base64}`}
                                  alt={`Preprocessed page ${frame.page_index}`}
                                />
                                <div className="frame-meta">
                                  <strong>Page {frame.page_index}</strong>
                                  <span>
                                    {frame.width} x {frame.height} to {frame.preprocessed_width} x {frame.preprocessed_height}
                                  </span>
                                  <span>{frame.mode}</span>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="empty-state compact">No preprocessed frames yet.</div>
                          )}
                        </div>
                      </article>
                    </section>
                  ) : null}

                  {result ? (
                    <section className="json-card compact">
                      <div className="card-header">
                        <div>
                          <div className="pill muted">Debug</div>
                          <h2>Raw payload</h2>
                        </div>
                      </div>
                      <pre>{JSON.stringify(result, null, 2)}</pre>
                    </section>
                  ) : null}
                </div>

                <div className="preview-card">
                  <div className="card-header">
                    <div>
                      <div className="pill muted">Preview</div>
                      <h2>Local source preview</h2>
                    </div>
                    <span className="hint">Client-side only</span>
                  </div>

                  {selectedPreviewUrl ? (
                    <div className="preview-wrapper">
                      <img className="preview-image" src={selectedPreviewUrl} alt="Selected upload preview" />
                      <div className="preview-overlay">
                        <span className="preview-badge">Original</span>
                      </div>
                    </div>
                  ) : (
                    <div className="empty-state">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <circle cx="15.5" cy="15.5" r="1.5" />
                        <circle cx="8.5" cy="15.5" r="1.5" />
                        <circle cx="15.5" cy="8.5" r="1.5" />
                      </svg>
                      <span>Image previews appear here.</span>
                      <p>PDFs are summarized after upload because the browser preview is text-free.</p>
                    </div>
                  )}
                </div>
              </section>
            </>
          )}

          {activeTab === 'results' && detection && (
            <>
              <section className="hero-panel compact-hero">
                <p className="lede">
                  Analysis results for {detection.filename} - Industry: {detection.industry || 'Unknown'}
                </p>
              </section>

              <section className="workspace">
                <article className="result-card">
                  <div className="card-header">
                    <div>
                      <div className="pill">Industry</div>
                      <h2>{detection.industry || 'Unknown'}</h2>
                    </div>
                  </div>
                  <p className="industry-description">
                    This P&ID diagram has been identified as belonging to the {detection.industry || 'Unknown'} industry.
                    Component detection results are displayed below.
                  </p>
                </article>

                <section className="json-card compact">
                  <div className="card-header">
                    <div>
                      <div className="pill muted">OpenRouter</div>
                      <h2>Component counts</h2>
                    </div>
                  </div>

                  <div className="detection-layout">
                    <div className="response-summary">
                      <div><span>Filename</span><strong>{detection.filename}</strong></div>
                      <div><span>Models</span><strong>{detection.models_used.join(' + ')}</strong></div>
                      <div><span>Pages</span><strong>{detection.page_count}</strong></div>
                      <div><span>Source</span><strong>{detection.source_type}</strong></div>
                    </div>

                    {detection.pages.map((page) => (
                      <article className="detection-page" key={page.page_index}>
                        <div className="card-header">
                          <div>
                            <div className="pill muted">Page {page.page_index}</div>
                            <h3>Counts by category</h3>
                          </div>
                        </div>

                        <div className="category-chips">
                          {(['motor', 'pump', 'tank', 'valve'] as const).map((category) => {
                            const count = page.counts[category]
                            return (
                              <span className="category-chip" key={category}>
                                {category} {count}
                              </span>
                            )
                          })}
                        </div>

                        <div className="count-grid">
                          {(['motor', 'pump', 'tank', 'valve'] as const).map((category) => (
                            <div className="count-card" key={category}>
                              <span>{category}</span>
                              <strong>{page.counts[category]}</strong>
                            </div>
                          ))}
                        </div>

                        <div className="count-note">Verified by {page.model_results.map((result) => result.model).join(' + ')}</div>
                      </article>
                    ))}
                  </div>
                </section>
              </section>
            </>
          )}

          {activeTab === 'coordinates' && coordinates && (
            <>
              <section className="hero-panel compact-hero">
                  <p className="lede">
                    Component coordinates for {currentFile?.name || 'uploaded file'} - {displayedCoordinateCount} components detected
                  </p>
                  <p className="lede">
                    Detected components total: {displayedDetectedComponents}<br />
                    Coordinate entries: {displayedCoordinateCount}<br />
                    Difference: {displayedDetectedComponents - displayedCoordinateCount}
                  </p>
                  {displayedDetectedComponents - displayedCoordinateCount > 0 && (
                    <div className="warning-banner">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                        <line x1="12" y1="9" x2="12" y2="13" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                      </svg>
                      <span>Warning: {displayedDetectedComponents - displayedCoordinateCount} component(s) missing coordinates.</span>
                    </div>
                  )}
                </section>

              {batchRuns.length > 1 ? (
                <section className="workspace">
                  <article className="result-card wide">
                    <div className="card-header">
                      <div>
                        <div className="pill muted">Batch</div>
                        <h2>Visible coordinates by file</h2>
                      </div>
                    </div>
                    <div className="frame-grid">
                      {batchRuns.map((run) => {
                        const visibleCount = run.coordinates?.root.children.length ?? 0
                        return (
                          <div className="frame-card" key={run.file.name}>
                            <div className="frame-meta">
                              <strong>{run.file.name}</strong>
                              <span>Industry: {run.detection?.industry || 'Unknown'}</span>
                              <span>{visibleCount} visible coordinates</span>
                              <span>
                                Components: {run.detection
                                  ? run.detection.pages.reduce((sum, page) => {
                                      const c = page.counts
                                      return sum + (c.motor ?? 0) + (c.pump ?? 0) + (c.tank ?? 0) + (c.valve ?? 0)
                                    }, 0)
                                  : 0}
                              </span>
                              <span>{run.error ? `Error: ${run.error}` : 'Ready'}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </article>
                </section>
              ) : null}

              <section className="workspace">
                <article className="result-card">
                  <div className="card-header">
                    <div>
                      <div className="pill">Coordinates</div>
                      <h2>Detected Components</h2>
                    </div>
                  </div>
                  <p className="industry-description">
                    AI-detected component positions with bounding boxes. Each component includes its name, type, and exact coordinates.
                  </p>
                </article>

                <section className="json-card compact">
                  <div className="card-header">
                    <div>
                      <div className="pill muted">Components</div>
                      <h2>Position Data</h2>
                    </div>
                  </div>

                  <div className="coordinates-grid">
                    {coordinates.root.children.map((component, index) => (
                      <article className="coordinate-card" key={index}>
                        <div className="coordinate-header">
                          <div className="pill component-type">{component.type.replace('ia.symbol.', '')}</div>
                          <strong>{component.meta.name}</strong>
                          <span className="component-icon"><ComponentIcon /></span>
                        </div>
                        <div className="coordinate-details">
                          <div className="coordinate-item">
                            <span>X</span>
                            <strong>{component.position.x}</strong>
                          </div>
                          <div className="coordinate-item">
                            <span>Y</span>
                            <strong>{component.position.y}</strong>
                          </div>
                          <div className="coordinate-item">
                            <span>Width</span>
                            <strong>{component.position.width}</strong>
                          </div>
                          <div className="coordinate-item">
                            <span>Height</span>
                            <strong>{component.position.height}</strong>
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
