const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || ''
  const data = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const message = typeof data === 'object' && data.detail ? data.detail : 'Request gagal.'
    throw new Error(message)
  }
  return data
}

export async function getDatasetStatus() {
  const response = await fetch(`${API_BASE_URL}/api/dataset/status`)
  return parseResponse(response)
}

export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}/api/dataset/upload`, {
    method: 'POST',
    body: formData,
  })
  return parseResponse(response)
}

export async function trainModel(payload) {
  const response = await fetch(`${API_BASE_URL}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE_URL}/api/metrics`)
  return parseResponse(response)
}

export async function predictMessage(payload) {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseResponse(response)
}

export async function batchPredict(file, threshold = 0.5) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}/api/batch-predict?threshold=${threshold}`, {
    method: 'POST',
    body: formData,
  })
  return parseResponse(response)
}

export const downloadUrl = (path) => `${API_BASE_URL}${path}`
