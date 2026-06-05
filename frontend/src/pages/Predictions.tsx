import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import { predictPlayerOutcome } from '../services/api'
import PredictionForm from '../components/PredictionForm'

export default function Predictions() {
  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [selectedCourse, setSelectedCourse] = useState('')
  const predictionApi = useApi(() =>
    predictPlayerOutcome(selectedPlayer, selectedCourse)
  )

  const handlePredict = () => {
    if (selectedPlayer && selectedCourse) {
      predictionApi.execute()
    }
  }

  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold">Player Predictions</h1>

      <PredictionForm
        onPlayerChange={setSelectedPlayer}
        onCourseChange={setSelectedCourse}
        onSubmit={handlePredict}
        loading={predictionApi.loading}
      />

      {predictionApi.data && (
        <div className="card">
          <h2 className="text-2xl font-semibold mb-6">
            Prediction Results: {selectedPlayer} @ {selectedCourse}
          </h2>

          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-gray-600 text-sm mb-2">Make Cut</p>
              <p className="text-3xl font-bold text-blue-600">
                {(predictionApi.data.outcome_probabilities.make_cut * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <p className="text-gray-600 text-sm mb-2">Top 10</p>
              <p className="text-3xl font-bold text-green-600">
                {(predictionApi.data.outcome_probabilities.top_10 * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <p className="text-gray-600 text-sm mb-2">Win</p>
              <p className="text-3xl font-bold text-purple-600">
                {(predictionApi.data.outcome_probabilities.win * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-gray-600">Model Confidence: {(predictionApi.data.confidence * 100).toFixed(1)}%</p>
            {predictionApi.data.fit_score !== null && (
              <p className="text-gray-600 mt-2">
                Course Fit Score: {predictionApi.data.fit_score.toFixed(2)} strokes
              </p>
            )}
          </div>
        </div>
      )}

      {predictionApi.error && (
        <div className="card bg-red-50 border border-red-200">
          <p className="text-red-700">Error: {predictionApi.error.message}</p>
        </div>
      )}
    </div>
  )
}