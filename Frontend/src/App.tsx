import { useEffect, useRef, useState, type DragEvent } from 'react'
import './App.css'

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
  text: number
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Stage1Response | null>(null)
  const [detection, setDetection] = useState<DetectionResponse | null>(null)
  const [coordinates, setCoordinates] = useState<CoordinateDetectionResponse | null>(null)
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'upload' | 'results' | 'coordinates'>('upload')

  const totalDetectedComponents = detection
    ? detection.pages.reduce((sum, page) => {
        const c = page.counts
        return sum + (c.text ?? 0) + (c.motor ?? 0) + (c.pump ?? 0) + (c.tank ?? 0) + (c.valve ?? 0)
      }, 0)
    : 0

  const supportsPreview = selectedFile
    ? selectedFile.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(selectedFile.name)
    : false

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file)
    setError(null)
    setResult(null)
    setDetection(null)
    setCoordinates(null)
    setActiveTab('upload')
  }

  const postFile = async (endpoint: string) => {
    if (!selectedFile) {
      throw new Error('Choose an image or PDF first.')
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

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
    setIsUploading(true)
    setError(null)
    try {
      const uploadPayload = (await postFile('/upload')) as Stage1Response
      setResult(uploadPayload)

      const detectionPayload = (await postFile('/detect')) as DetectionResponse
      setDetection(detectionPayload)

      const coordinatesPayload = (await postFile('/coordinates')) as CoordinateDetectionResponse
      setCoordinates(coordinatesPayload)
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
    if (result && detection && coordinates) {
      const combined = JSON.stringify({ result, detection, coordinates }, null, 2)
      const blob = new Blob([combined], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      setDownloadUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setDownloadUrl(null)
    }
  }, [result, detection, coordinates])

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    const droppedFile = event.dataTransfer.files[0] ?? null
    handleFileSelect(droppedFile)
  }

  useEffect(() => {
    if (!selectedFile || !supportsPreview) {
      setSelectedPreviewUrl(null)
      return
    }

    const previewUrl = URL.createObjectURL(selectedFile)
    setSelectedPreviewUrl(previewUrl)

    return () => URL.revokeObjectURL(previewUrl)
  }, [selectedFile, supportsPreview])

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <h2>P&ID Dashboard</h2>
        </div>

        <nav className="nav">
          <button
            className="primary-button"
            onClick={runAll}
            disabled={isUploading || !selectedFile}
          >
            {isUploading ? 'Processing...' : 'Run All'}
          </button>
          {downloadUrl && (
            <a href={downloadUrl} download="output.json" className="secondary-button">
              Download JSON
            </a>
          )}
          <button 
            className={`nav-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload
          </button>
          <button 
            className={`nav-item ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
            disabled={!detection}
          >
            Analysis
          </button>
          <button 
            className={`nav-item ${activeTab === 'coordinates' ? 'active' : ''}`}
            onClick={() => setActiveTab('coordinates')}
            disabled={!coordinates}
          >
            Coordinates
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
          </div>
          <div className="topbar-right">
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
                  </div>

                  <div className="upload-controls">
                    <input
                      ref={fileInputRef}
                      className="file-input"
                      type="file"
                      accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf"
                      onChange={(event) => handleFileSelect(event.target.files?.[0] ?? null)}
                    />
                    <button type="button" className="secondary-button" onClick={() => fileInputRef.current?.click()}>
                      Browse files
                    </button>

                  </div>

                  <div className="selection-row">
                    <div>
                      <span className="label">Selected file</span>
                      <strong>{selectedFile?.name ?? 'None yet'}</strong>
                    </div>
                    <div>
                      <span className="label">Type</span>
                      <strong>{selectedFile?.type || 'Unknown'}</strong>
                    </div>
                    <div>
                      <span className="label">Size</span>
                      <strong>{selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : '--'}</strong>
                    </div>
                  </div>

                  {error ? <div className="status error">{error}</div> : null}
                  {!error && result && !detection ? <div className="status success">Stage 1 completed successfully. Ready for AI analysis.</div> : null}
                  {!error && detection ? <div className="status success">AI detection completed! Industry identified: {detection.industry || 'Unknown'}</div> : null}
                  {!error && coordinates ? <div className="status success">Coordinate detection completed! {coordinates.root.children.length} components found.</div> : null}

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
                    <img className="preview-image" src={selectedPreviewUrl} alt="Selected upload preview" />
                  ) : (
                    <div className="empty-state">
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
                          {(['text', 'motor', 'pump', 'tank', 'valve'] as const).map((category) => {
                            const count = page.counts[category]
                            return (
                              <span className="category-chip" key={category}>
                                {category} {count}
                              </span>
                            )
                          })}
                        </div>

                        <div className="count-grid">
                          {(['text', 'motor', 'pump', 'tank', 'valve'] as const).map((category) => (
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
                    Component coordinates for {selectedFile?.name || 'uploaded file'} - {coordinates.root.children.length} components detected
                  </p>
                  <p className="lede">
                    Detected components total: {totalDetectedComponents}<br />
                    Coordinate entries: {coordinates.root.children.length}<br />
                    Difference: {totalDetectedComponents - coordinates.root.children.length}
                  </p>
                  {totalDetectedComponents - coordinates.root.children.length > 0 && (
                    <p className="warning" style={{ color: 'orange' }}>
                      Warning: {totalDetectedComponents - coordinates.root.children.length} component(s) missing coordinates.
                    </p>
                  )}
                </section>

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
