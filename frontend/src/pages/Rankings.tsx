import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import { getTournamentRankings } from '../services/api'

export default function Rankings() {
  const [tournament, setTournament] = useState('masters_2024')
  const [courses, setCourses] = useState(['augusta_national'])
  const rankingsApi = useApi(() =>
    getTournamentRankings(tournament, courses, 50)
  )

  const handleGetRankings = () => {
    rankingsApi.execute()
  }

  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold">Tournament Rankings</h1>

      <div className="card">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">Tournament</label>
            <input
              type="text"
              value={tournament}
              onChange={(e) => setTournament(e.target.value)}
              className="w-full border rounded px-3 py-2"
              placeholder="e.g., masters_2024"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Course(s)</label>
            <input
              type="text"
              value={courses.join(', ')}
              onChange={(e) => setCourses(e.target.value.split(',').map(c => c.trim()))}
              className="w-full border rounded px-3 py-2"
              placeholder="e.g., augusta_national"
            />
          </div>
        </div>
        <button onClick={handleGetRankings} className="btn-primary">
          {rankingsApi.loading ? 'Loading...' : 'Get Rankings'}
        </button>
      </div>

      {rankingsApi.data?.aggregate_ranking && (
        <div className="card">
          <h2 className="text-2xl font-semibold mb-4">Top Players</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-4 py-2 text-left">Rank</th>
                  <th className="px-4 py-2 text-left">Player</th>
                  <th className="px-4 py-2 text-right">Fit Score</th>
                  <th className="px-4 py-2 text-right">Courses</th>
                </tr>
              </thead>
              <tbody>
                {rankingsApi.data.aggregate_ranking.slice(0, 25).map((player: any) => (
                  <tr key={player.rank} className="border-b hover:bg-gray-50">
                    <td className="px-4 py-2 font-semibold">{player.rank}</td>
                    <td className="px-4 py-2">{player.player_id}</td>
                    <td className="px-4 py-2 text-right">
                      {player.average_fit_score.toFixed(2)}
                    </td>
                    <td className="px-4 py-2 text-right">{player.courses_analyzed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {rankingsApi.error && (
        <div className="card bg-red-50 border border-red-200">
          <p className="text-red-700">Error: {rankingsApi.error.message}</p>
        </div>
      )}
    </div>
  )
}