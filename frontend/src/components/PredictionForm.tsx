interface Props {
  onPlayerChange: (player: string) => void
  onCourseChange: (course: string) => void
  onSubmit: () => void
  loading: boolean
}

export default function PredictionForm({
  onPlayerChange,
  onCourseChange,
  onSubmit,
  loading,
}: Props) {
  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-6">Make a Prediction</h2>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium mb-2">Player</label>
          <input
            type="text"
            placeholder="e.g., scheffler_scottie"
            onChange={(e) => onPlayerChange(e.target.value)}
            className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Course</label>
          <input
            type="text"
            placeholder="e.g., augusta_national"
            onChange={(e) => onCourseChange(e.target.value)}
            className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
      <button onClick={onSubmit} disabled={loading} className="btn-primary disabled:opacity-50">
        {loading ? 'Predicting...' : 'Get Prediction'}
      </button>
    </div>
  )
}