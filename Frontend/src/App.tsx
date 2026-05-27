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
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Stage1Response | null>(null)
  const [detection, setDetection] = useState<DetectionResponse | null>(null)
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState<string | null>(null)

  const supportsPreview = selectedFile
    ? selectedFile.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(selectedFile.name)
    : false

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file)
    setError(null)
    setResult(null)
    setDetection(null)
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

  const submitUpload = async () => {
    setIsUploading(true)
    setError(null)

    try {
      const payload = (await postFile('/upload')) as Stage1Response
      setResult(payload)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Upload failed.')
    } finally {
      setIsUploading(false)
    }
  }

  const submitDetection = async () => {
    setIsUploading(true)
    setError(null)

    try {
      const payload = (await postFile('/detect')) as DetectionResponse
      setDetection(payload)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Detection failed.')
    } finally {
      setIsUploading(false)
    }
  }

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
    <main className="shell">
      <section className="hero-panel">
        <div className="eyebrow">Stage 1 - Image Input</div>
        <h1>P&ID upload workspace</h1>
        <p className="lede">
          Drop a P&ID image or PDF, send it to the backend, and inspect the loaded
          frames before detection starts.
        </p>

        <div className="meta-row">
          <span>Backend: {API_BASE_URL}</span>
          <span>Accepts PNG, JPG, WEBP, BMP, TIFF, PDF</span>
        </div>
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
            <button
              type="button"
              className="primary-button"
              onClick={submitUpload}
              disabled={isUploading || !selectedFile}
            >
              {isUploading ? 'Uploading...' : 'Run Stage 1'}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={submitDetection}
              disabled={isUploading || !selectedFile}
            >
              {isUploading ? 'Working...' : 'Run AI detection'}
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
          {!error && result && !detection ? <div className="status success">Stage 1 completed successfully.</div> : null}
          {!error && detection ? <div className="status success">AI detection completed!</div> : null}

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

          {detection ? (
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
    </main>
  )
}

export default App
