import { useEffect, useRef, useState } from 'react'
import { Icon } from './Icons'

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
        <button type="button" className="btn-secondary w-full" onClick={start}><Icon name="camera" className="h-4 w-4" />Use webcam</button>
      ) : (
        <div className="space-y-3 animate-fade-up">
          <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-md bg-black" aria-label="Camera preview" />
          <div className="flex gap-2">
            <button type="button" className="btn-primary flex-1" onClick={capture}><Icon name="camera" className="h-4 w-4" />Capture photo</button>
            <button type="button" className="btn-secondary" onClick={stop}>Cancel</button>
          </div>
        </div>
      )}
      {error && <p role="alert" className="rounded-md bg-bad-tint px-3 py-2 text-xs text-bad">{error}</p>}
    </div>
  )
}
