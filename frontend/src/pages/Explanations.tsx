import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import { getExplanation, getFeatureImportance } from '../services/api'

export default function Explanations() {
  const [playerID, setPlayerID] = useState('scheffler_scottie')
  const [courseID, setCourseID] = useState('augusta_national')
  const explanationApi = useApi(() =>
    getExplanation(playerID, courseID, 10)
  )
  const importanceApi = useApi(() => getFeatureImportance(15))

  const handleExplain = () => {
    explanationApi.execute()
  }

  const handleGetImportance = () => {
    importanceApi.execute()
  }

  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold">Model Explanations</h1>

      {/* Local Explanation */}
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold">Local Prediction Explanation</h2>
        <div className="card">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium mb-2">Player</label>
              <input
                type="text"
                value={playerID}
                onChange={(e) => setPlayerID(e.target.value)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Course</label>
              <input
                type="text"
                value={courseID}
                onChange={(e) => setCourseID(e.target.value)}
                className="w-full border rounded px-3 py-2"
              />
            </div>
          </div>
          <button onClick={handleExplain} className="btn-primary">
            {explanationApi.loading ? 'Loading...' : 'Explain Prediction'}
          </button>
        </div>

        {explanationApi.data?.local_explanation?.top_features && (
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Top Contributing Features</h3>
            <div className="space-y-3">
              {explanationApi.data.local_explanation.top_features.map(
                (feature: any, i: number) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="font-medium">{feature.feature}</span>
                    <span className="text-sm text-gray-600">
                      {feature.importance.toFixed(4)}
                    </span>
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>

      {/* Global Feature Importance */}
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold">Global Feature Importance</h2>
        <button onClick={handleGetImportance} className="btn-primary">
          {importanceApi.loading ? 'Loading...' : 'Get Feature Importance'}
        </button>

        {importanceApi.data && Array.isArray(importanceApi.data) && (
          <div className="card">
            <div className="space-y-3">
              {importanceApi.data.map((feature: any, i: number) => (
                <div key={i}>
                  <div className="flex justify-between mb-1">
                    <span>{feature.feature}</span>
                    <span className="text-sm text-gray-600">
                      {(feature.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${feature.importance * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}