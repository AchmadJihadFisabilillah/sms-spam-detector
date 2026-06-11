import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  FileUp,
  FlaskConical,
  Loader2,
  MailWarning,
  Play,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import {
  batchPredict,
  downloadUrl,
  getDatasetStatus,
  getMetrics,
  predictMessage,
  trainModel,
  uploadDataset,
} from './api'

const formatPct = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`
const formatNum = (value) => Number(value || 0).toLocaleString('en-US')
const classNames = (...classes) => classes.filter(Boolean).join(' ')

function Section({ title, subtitle, icon: Icon, children, action }) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex gap-3">
          {Icon ? (
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-slate-900 text-white">
              <Icon size={22} />
            </div>
          ) : null}
          <div>
            <h2 className="text-xl font-bold text-slate-950">{title}</h2>
            {subtitle ? <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">{subtitle}</p> : null}
          </div>
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

function Pill({ children, tone = 'slate' }) {
  const styles = {
    slate: 'bg-slate-100 text-slate-700 ring-slate-200',
    green: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    red: 'bg-rose-50 text-rose-700 ring-rose-200',
    amber: 'bg-amber-50 text-amber-700 ring-amber-200',
    blue: 'bg-blue-50 text-blue-700 ring-blue-200',
  }
  return <span className={classNames('rounded-full px-3 py-1 text-xs font-semibold ring-1', styles[tone])}>{children}</span>
}

function MetricCard({ label, value, hint }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-black text-slate-950">{value}</p>
      {hint ? <p className="mt-1 text-xs text-slate-500">{hint}</p> : null}
    </div>
  )
}

function ProgressBar({ value, label, tone = 'blue' }) {
  const percent = Math.max(0, Math.min(100, Number(value || 0) * 100))
  const styles = {
    blue: 'bg-blue-600',
    green: 'bg-emerald-600',
    red: 'bg-rose-600',
    amber: 'bg-amber-500',
    slate: 'bg-slate-800',
  }
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs font-semibold text-slate-600">
        <span>{label}</span>
        <span>{percent.toFixed(2)}%</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-200">
        <div className={classNames('h-full rounded-full transition-all', styles[tone])} style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix }) {
  if (!matrix) return null
  const [[tn, fp], [fn, tp]] = matrix
  const cells = [
    { label: 'True Negative', value: tn, hint: 'ham benar' },
    { label: 'False Positive', value: fp, hint: 'ham salah jadi spam' },
    { label: 'False Negative', value: fn, hint: 'spam lolos' },
    { label: 'True Positive', value: tp, hint: 'spam benar' },
  ]
  return (
    <div className="grid grid-cols-2 gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-3">
      {cells.map((cell) => (
        <div key={cell.label} className="rounded-xl bg-white p-3 text-center shadow-sm">
          <p className="text-xs font-semibold text-slate-500">{cell.label}</p>
          <p className="mt-1 text-2xl font-black text-slate-950">{formatNum(cell.value)}</p>
          <p className="text-[11px] text-slate-500">{cell.hint}</p>
        </div>
      ))}
    </div>
  )
}

function DatasetPanel({ status, onRefresh, onUpload, loading }) {
  const [file, setFile] = useState(null)
  const hasDataset = status?.dataset_exists && status?.rows
  return (
    <Section
      icon={Database}
      title="1. Dataset asli Kaggle"
      subtitle="Gunakan SMS Spam Collection Dataset dari Kaggle. Upload file spam.csv dengan kolom v1 sebagai label dan v2 sebagai isi SMS."
      action={<button onClick={onRefresh} className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Refresh</button>}
    >
      <div className="grid gap-4 lg:grid-cols-3">
        <MetricCard label="Status Dataset" value={hasDataset ? 'Valid' : 'Belum siap'} hint={status?.message || 'Mengecek dataset...'} />
        <MetricCard label="Total Data" value={hasDataset ? formatNum(status.rows) : '-'} hint="setelah cleaning dan deduplikasi" />
        <MetricCard label="Distribusi Label" value={status?.label_distribution ? `${status.label_distribution.ham || 0} ham / ${status.label_distribution.spam || 0} spam` : '-'} hint="target klasifikasi" />
      </div>

      <div className="mt-5 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-bold text-slate-900">Upload dataset Kaggle</p>
            <p className="mt-1 text-sm text-slate-600">Pilih file <code className="rounded bg-white px-1">spam.csv</code>. Dataset tidak dibuat manual; data harus berasal dari Kaggle.</p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">
              <FileUp size={18} />
              Pilih CSV
              <input className="hidden" type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            </label>
            <button
              disabled={!file || loading}
              onClick={() => file && onUpload(file)}
              className="rounded-xl bg-slate-950 px-4 py-2 text-sm font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {loading ? 'Uploading...' : 'Upload'}
            </button>
          </div>
        </div>
        {file ? <p className="mt-3 text-sm text-slate-600">File dipilih: <b>{file.name}</b></p> : null}
      </div>
    </Section>
  )
}

function TrainingPanel({ onTrain, training, metrics }) {
  const [testSize, setTestSize] = useState(0.2)
  const [maxFeatures, setMaxFeatures] = useState(8000)
  return (
    <Section
      icon={FlaskConical}
      title="2. Training model"
      subtitle="Backend akan melatih dua model: Multinomial Naive Bayes dan Logistic Regression, lalu menyimpan model, metrik, confusion matrix, dan hasil McNemar Test."
    >
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr_auto] lg:items-end">
        <label className="block">
          <span className="text-sm font-semibold text-slate-700">Test size</span>
          <select value={testSize} onChange={(event) => setTestSize(Number(event.target.value))} className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-slate-900">
            <option value={0.2}>20% testing / 80% training</option>
            <option value={0.25}>25% testing / 75% training</option>
            <option value={0.3}>30% testing / 70% training</option>
          </select>
        </label>
        <label className="block">
          <span className="text-sm font-semibold text-slate-700">Max TF-IDF features</span>
          <select value={maxFeatures} onChange={(event) => setMaxFeatures(Number(event.target.value))} className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none focus:border-slate-900">
            <option value={5000}>5.000</option>
            <option value={8000}>8.000</option>
            <option value={12000}>12.000</option>
            <option value={20000}>20.000</option>
          </select>
        </label>
        <button
          onClick={() => onTrain({ test_size: testSize, random_state: 42, max_features: maxFeatures })}
          disabled={training}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 font-bold text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-300"
        >
          {training ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
          {training ? 'Training...' : 'Train Model'}
        </button>
      </div>

      {metrics ? (
        <div className="mt-5 grid gap-4 lg:grid-cols-4">
          <MetricCard label="Training rows" value={formatNum(metrics.dataset.train_rows)} />
          <MetricCard label="Testing rows" value={formatNum(metrics.dataset.test_rows)} />
          <MetricCard label="Best model" value={metrics.best_model === 'naive_bayes' ? 'Naive Bayes' : 'Logistic Regression'} />
          <MetricCard label="Trained at" value={new Date(metrics.trained_at).toLocaleString()} />
        </div>
      ) : null}
    </Section>
  )
}

function EvaluationPanel({ metrics }) {
  if (!metrics) {
    return (
      <Section icon={BarChart3} title="3. Evaluasi dan perbandingan" subtitle="Metrik akan muncul setelah training selesai.">
        <div className="rounded-2xl bg-slate-50 p-6 text-center text-slate-600">Belum ada hasil training.</div>
      </Section>
    )
  }

  return (
    <Section
      icon={BarChart3}
      title="3. Evaluasi dan perbandingan model"
      subtitle="Model dibandingkan menggunakan accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, Log Loss, Brier Score, weighted ranking, dan McNemar Test."
      action={
        <div className="flex gap-2">
          <a href={downloadUrl('/api/download/metrics')} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"><Download size={16} /> Metrics</a>
          <a href={downloadUrl('/api/download/predictions')} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"><Download size={16} /> Predictions</a>
        </div>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        {metrics.models.map((model) => (
          <div key={model.model_key} className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="text-lg font-black text-slate-950">{model.model_name}</h3>
                <p className="text-sm text-slate-600">Final score: <b>{formatPct(model.final_score)}</b></p>
              </div>
              {metrics.best_model === model.model_key ? <Pill tone="green">Best model</Pill> : <Pill>Comparison</Pill>}
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <ProgressBar label="Accuracy" value={model.accuracy} tone="blue" />
              <ProgressBar label="Precision" value={model.precision} tone="green" />
              <ProgressBar label="Recall" value={model.recall} tone="amber" />
              <ProgressBar label="F1-score" value={model.f1_score} tone="red" />
              <ProgressBar label="ROC-AUC" value={model.roc_auc} tone="slate" />
              <ProgressBar label="PR-AUC" value={model.pr_auc} tone="blue" />
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <MetricCard label="Log Loss" value={Number(model.log_loss).toFixed(4)} hint="lebih kecil lebih baik" />
              <MetricCard label="Brier Score" value={Number(model.brier_score).toFixed(4)} hint="lebih kecil lebih baik" />
            </div>
            <div className="mt-4">
              <ConfusionMatrix matrix={model.confusion_matrix} />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 rounded-3xl border border-slate-200 bg-white p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="text-lg font-black text-slate-950">McNemar Test</h3>
            <p className="mt-1 text-sm text-slate-600">Uji statistik untuk mengetahui apakah perbedaan dua model signifikan atau hanya kebetulan.</p>
          </div>
          <Pill tone={metrics.mcnemar_test.is_significant ? 'green' : 'amber'}>
            p-value {Number(metrics.mcnemar_test.p_value).toFixed(6)}
          </Pill>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-4">
          <MetricCard label="NB benar, LR benar" value={formatNum(metrics.mcnemar_test.table[0][0])} />
          <MetricCard label="NB benar, LR salah" value={formatNum(metrics.mcnemar_test.table[0][1])} />
          <MetricCard label="NB salah, LR benar" value={formatNum(metrics.mcnemar_test.table[1][0])} />
          <MetricCard label="NB salah, LR salah" value={formatNum(metrics.mcnemar_test.table[1][1])} />
        </div>
        <div className="mt-4 rounded-2xl bg-slate-50 p-4 text-sm font-semibold text-slate-700">{metrics.mcnemar_test.interpretation}</div>
      </div>
    </Section>
  )
}
function PredictionPanel({ metrics }) {
  const [message, setMessage] = useState('Congratulations! You won a free cash prize. Claim now!')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const threshold = 0.5

  async function handlePredict() {
    setError(null)
    setLoading(true)
    try {
      setResult(await predictMessage({ message, threshold }))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Section
      icon={MailWarning}
      title="Deteksi SMS Baru"
      subtitle="Masukkan pesan SMS, sistem akan menghitung probabilitas spam dengan dua model dan mengambil keputusan berdasarkan model terbaik."
    >
      <div className="grid gap-5 lg:grid-cols-[1fr_420px]">
        <div>
          <label className="block text-sm font-semibold text-slate-700">Teks SMS</label>
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            rows={7}
            className="mt-2 w-full rounded-2xl border border-slate-300 bg-white p-4 text-slate-900 outline-none focus:border-slate-900"
            placeholder="Contoh: URGENT! Your number has won a reward. Call now."
          />
          <div className="mt-4 flex justify-between items-center">
            <span className="text-sm font-semibold text-slate-500">Threshold default: {threshold.toFixed(2)}</span>
            <button onClick={handlePredict} disabled={loading || !message.trim()} className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-5 py-3 font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-400">
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
              Prediksi
            </button>
          </div>
          {error ? <div className="mt-4 rounded-xl bg-rose-50 p-4 text-sm font-semibold text-rose-700">{error}</div> : null}
        </div>

        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
          {!result ? (
            <div className="flex h-full min-h-72 flex-col items-center justify-center text-center text-slate-500">
              {metrics ? <ShieldAlert size={40} /> : <AlertCircle size={40} />}
              <p className="mt-3 font-semibold">{metrics ? 'Hasil prediksi akan muncul di sini.' : 'Latih model terlebih dahulu sebelum prediksi.'}</p>
            </div>
          ) : (
            <div>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-500">Final Prediction</p>
                  <h3 className={classNames('mt-1 text-3xl font-black', result.final_prediction_label === 'spam' ? 'text-rose-700' : 'text-emerald-700')}>
                    {result.final_prediction_label === 'spam' ? 'SPAM' : 'HAM'}
                  </h3>
                </div>
                {result.final_prediction_label === 'spam' ? <ShieldAlert className="text-rose-600" size={36} /> : <ShieldCheck className="text-emerald-600" size={36} />}
              </div>
              
              {result.translated_message && result.translated_message.toLowerCase().trim() !== result.raw_message.toLowerCase().trim() && (
                <div className="mt-3 rounded-2xl bg-blue-50 border border-blue-100 p-4 text-sm text-blue-900 animate-fadeIn">
                  <span className="font-bold block mb-1">Terjemahan Otomatis (Inggris):</span>
                  <p className="italic">"{result.translated_message}"</p>
                </div>
              )}

              <div className="mt-4 space-y-3">
                <ProgressBar label="Final spam probability" value={result.final_spam_probability} tone={result.final_prediction_label === 'spam' ? 'red' : 'green'} />
                <MetricCard label="Risk level" value={result.risk_level} hint={`Best model: ${result.best_model || 'average ensemble'}`} />
              </div>
              <div className="mt-4 space-y-3">
                {result.predictions.map((item) => (
                  <div key={item.model_name} className="rounded-2xl bg-white p-4 shadow-sm">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <p className="font-bold text-slate-900">{item.model_name}</p>
                      <Pill tone={item.prediction_label === 'spam' ? 'red' : 'green'}>{item.prediction_label}</Pill>
                    </div>
                    <ProgressBar label="P(spam)" value={item.spam_probability} tone={item.prediction_label === 'spam' ? 'red' : 'green'} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Section>
  )
}



function BatchPanel() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleBatch() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await batchPredict(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Section icon={FileUp} title="5. Batch prediction" subtitle="Upload CSV berisi kolom message, text, sms, atau v2 untuk memprediksi banyak SMS sekaligus.">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50">
          <FileUp size={18} />
          Pilih CSV batch
          <input className="hidden" type="file" accept=".csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
        </label>
        <button onClick={handleBatch} disabled={!file || loading} className="rounded-xl bg-slate-950 px-5 py-3 font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-400">
          {loading ? 'Memproses...' : 'Prediksi Batch'}
        </button>
        {result?.download_url ? <a href={downloadUrl(result.download_url)} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50"><Download size={16} /> Download hasil</a> : null}
      </div>
      {file ? <p className="mt-3 text-sm text-slate-600">File dipilih: <b>{file.name}</b></p> : null}
      {error ? <div className="mt-4 rounded-xl bg-rose-50 p-4 text-sm font-semibold text-rose-700">{error}</div> : null}
      {result?.preview ? (
        <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200">
          <table className="w-full min-w-[700px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="p-3">Message</th>
                <th className="p-3">Prediction</th>
                <th className="p-3">P(spam)</th>
                <th className="p-3">Risk</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {result.preview.map((row, index) => (
                <tr key={index}>
                  <td className="p-3 text-slate-700">{row.message}</td>
                  <td className="p-3 font-bold">{row.final_prediction}</td>
                  <td className="p-3">{formatPct(row.final_spam_probability)}</td>
                  <td className="p-3">{row.risk_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Section>
  )
}

function MethodologyPanel() {
  return (
    <Section icon={Sparkles} title="6. Metodologi project akhir" subtitle="Bagian ini bisa langsung kamu pindahkan ke laporan atau presentasi akhir.">
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl bg-slate-50 p-5">
          <h3 className="font-black text-slate-950">Input</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">Dataset Kaggle SMS Spam Collection dengan kolom label ham/spam dan teks SMS mentah.</p>
        </div>
        <div className="rounded-2xl bg-slate-50 p-5">
          <h3 className="font-black text-slate-950">Process</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">Cleaning teks, feature engineering, TF-IDF unigram/bigram, train-test split, training dua algoritma, dan evaluasi.</p>
        </div>
        <div className="rounded-2xl bg-slate-50 p-5">
          <h3 className="font-black text-slate-950">Output</h3>
          <p className="mt-2 text-sm leading-6 text-slate-600">Prediksi ham/spam, probabilitas spam, risk level, confusion matrix, ranking model, dan McNemar Test.</p>
        </div>
      </div>
      <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="font-black text-slate-950">Alur sistem</h3>
        <div className="mt-4 grid gap-3 text-sm font-semibold text-slate-700 sm:grid-cols-2 lg:grid-cols-5">
          {['Kaggle Dataset', 'Preprocessing', 'TF-IDF', 'NB + LR Training', 'Evaluation + Prediction'].map((step, index) => (
            <div key={step} className="rounded-xl bg-slate-50 p-4 text-center">
              <p className="text-xs text-slate-400">Step {index + 1}</p>
              <p className="mt-1">{step}</p>
            </div>
          ))}
        </div>
      </div>
    </Section>
  )
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [toast, setToast] = useState(null)

  async function refreshStatus() {
    try {
      setStatus(await getDatasetStatus())
    } catch (err) {
      setToast({ type: 'error', message: err.message })
    }
  }

  async function refreshMetrics() {
    try {
      setMetrics(await getMetrics())
    } catch (_) {
      setMetrics(null)
    }
  }

  useEffect(() => {
    refreshStatus()
    refreshMetrics()
  }, [])

  async function handleUpload(file) {
    setLoading(true)
    setToast(null)
    try {
      const nextStatus = await uploadDataset(file)
      setStatus(nextStatus)
      setToast({ type: 'success', message: 'Dataset berhasil diupload dan valid.' })
    } catch (err) {
      setToast({ type: 'error', message: err.message })
    } finally {
      setLoading(false)
    }
  }

  async function handleTrain(payload) {
    setTraining(true)
    setToast(null)
    try {
      const result = await trainModel(payload)
      setMetrics(result.metrics)
      await refreshStatus()
      setToast({ type: 'success', message: result.message })
    } catch (err) {
      setToast({ type: 'error', message: err.message })
    } finally {
      setTraining(false)
    }
  }

  const heroStats = useMemo(() => {
    if (!metrics) return null
    const best = metrics.models.find((item) => item.model_key === metrics.best_model)
    return { best }
  }, [metrics])

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-950 text-white">
                <ShieldAlert size={24} />
              </div>
              <div>
                <p className="text-sm font-bold uppercase tracking-wide text-blue-600">Final Project Machine Learning</p>
                <h1 className="text-2xl font-black text-slate-950">SMS Spam Detector</h1>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <Pill tone="blue">React + Tailwind</Pill>
              <Pill tone="green">FastAPI</Pill>
              <Pill>Naive Bayes</Pill>
              <Pill>Logistic Regression</Pill>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-5 py-8">
        <section className="overflow-hidden rounded-[2rem] bg-slate-950 p-8 text-white shadow-xl">
          <div className="grid gap-8 lg:grid-cols-[1.3fr_0.7fr] lg:items-center">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.25em] text-blue-300">Deteksi SMS Spam</p>
              <h2 className="mt-3 max-w-4xl text-4xl font-black leading-tight lg:text-5xl">
                Deteksi SMS spam secara real-time.
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-300">
                Masukkan pesan SMS untuk menganalisis probabilitas spam menggunakan model Naive Bayes dan Logistic Regression. Sistem secara otomatis mendeteksi bahasa non-Inggris dan menerjemahkannya agar pendeteksian berjalan akurat.
              </p>
              <div className="mt-6 flex flex-wrap gap-2">
                <Pill tone="green">Translasi Otomatis</Pill>
                <Pill tone="blue">Probability Prediction</Pill>
                <Pill tone="amber">Dual ML Models</Pill>
              </div>
            </div>
            <div className="rounded-3xl bg-white/10 p-5 backdrop-blur">
              <div className="grid gap-3">
                <MetricCard label="Dataset" value={status?.rows ? formatNum(status.rows) : '-'} hint="SMS valid" />
                <MetricCard label="Best model" value={metrics?.best_model ? (metrics.best_model === 'naive_bayes' ? 'Naive Bayes' : 'Logistic Regression') : '-'} hint="berdasarkan final score" />
                <MetricCard label="Best F1-score" value={heroStats?.best ? formatPct(heroStats.best.f1_score) : '-'} hint="setelah training" />
              </div>
            </div>
          </div>
        </section>

        {toast ? (
          <div className={classNames('flex items-center gap-3 rounded-2xl p-4 text-sm font-semibold', toast.type === 'success' ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700')}>
            {toast.type === 'success' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
            {toast.message}
          </div>
        ) : null}

        <PredictionPanel metrics={metrics} />
      </main>
    </div>
  )
}
