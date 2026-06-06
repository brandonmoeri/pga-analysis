import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import { predictPlayerOutcome } from '../services/api'
import PredictionForm from '../components/PredictionForm'
import ProbabilityBar from '../components/ProbabilityBar'

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
          {/* Header — matches CLI separator style */}
          <div className="font-mono text-gray-400 text-xs mb-1">{'='.repeat(50)}</div>
          <div className="font-mono text-sm font-bold text-gray-700 mb-1">PLAYER OUTCOME PREDICTION</div>
          <div className="font-mono text-gray-400 text-xs mb-4">{'='.repeat(50)}</div>

          <div className="font-mono text-sm text-gray-600 mb-6 space-y-1">
            <p>Player:  <span className="text-gray-900 font-semibold">{selectedPlayer}</span></p>
            <p>Course:  <span className="text-gray-900 font-semibold">{selectedCourse}</span></p>
            {predictionApi.data.fit_score != null && (
              <p>Fit Score: <span className="text-gray-900 font-semibold">{predictionApi.data.fit_score.toFixed(2)} strokes</span></p>
            )}
            <p>Confidence: <span className="text-gray-900 font-semibold">{(predictionApi.data.confidence * 100).toFixed(1)}%</span></p>
          </div>

          {predictionApi.data.course_history && (
            <>
              <div className="font-mono text-xs text-gray-400 mb-2">Course History:</div>
              <div className="font-mono text-sm text-gray-600 mb-6 space-y-1">
                <p>
                  SG Avg:{' '}
                  <span className={`font-semibold ${predictionApi.data.course_history.sg_avg >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                    {predictionApi.data.course_history.sg_avg >= 0 ? '+' : ''}
                    {predictionApi.data.course_history.sg_avg.toFixed(2)}
                  </span>
                </p>
                <p>
                  Appearances:{' '}
                  <span className="text-gray-900 font-semibold">
                    {predictionApi.data.course_history.appearances}
                  </span>
                </p>
              </div>
            </>
          )}

          <div className="font-mono text-xs text-gray-400 mb-4">Tournament Outcomes:</div>

          <div className="space-y-1">
            <ProbabilityBar
              label="Make Cut"
              probability={predictionApi.data.outcome_probabilities.make_cut}
              color="green"
            />
            <ProbabilityBar
              label="Top 10"
              probability={predictionApi.data.outcome_probabilities.top_10}
              color="yellow"
            />
            <ProbabilityBar
              label="Win"
              probability={predictionApi.data.outcome_probabilities.win}
              color="red"
            />
          </div>

          <div className="font-mono text-gray-400 text-xs mt-4">{'='.repeat(50)}</div>
        </div>
      )}

      {predictionApi.error && (
        <div className="card bg-red-50 border border-red-200">
          <p className="text-red-700 font-mono text-sm">Error: {predictionApi.error.message}</p>
        </div>
      )}
    </div>
  )
}