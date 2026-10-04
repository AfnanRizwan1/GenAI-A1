import { useEffect, useRef, useState } from 'react'

/** Opens the camera, shows a live preview and returns a captured PNG File through onCapture. */
export default function WebcamCapture({ onCapture }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [active, setActive] = useState(false)
  const [error, setError] = useState(null)

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setActive(false)
  }
  useEffect(() => stop, [])

  const start = async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser cannot access a camera (it needs HTTPS or localhost).')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      streamRef.current = stream
      setActive(true)
      requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = stream })
    } catch (e) {
      setError(e?.name === 'NotAllowedError' ? 'Camera permission was denied.' : 'Could not start the camera.')
    }
  }

  const capture = () => {
    const v = videoRef.current
    if (!v || !v.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = v.videoWidth
    canvas.height = v.videoHeight
    canvas.getContext('2d').drawImage(v, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) onCapture(new File([blob], 'webcam.png', { type: 'image/png' }))
      stop()
    }, 'image/png')
  }

  return (
    <div className="space-y-3">
      {!active ? (
        <button type="button" className="btn-secondary" onClick={start}>📷 Use webcam</button>
      ) : (
        <div className="space-y-2">
          <video ref={videoRef} autoPlay playsInline muted className="w-full max-w-sm rounded-xl bg-black" aria-label="Camera preview" />
          <div className="flex gap-2">
            <button type="button" className="btn-primary" onClick={capture}>Capture photo</button>
            <button type="button" className="btn-secondary" onClick={stop}>Cancel</button>
          </div>
        </div>
      )}
      {error && <p role="alert" className="text-sm text-red-600 dark:text-red-400">{error}</p>}
    </div>
  )
}
