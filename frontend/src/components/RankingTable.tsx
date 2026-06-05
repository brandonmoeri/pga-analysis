interface Player {
  rank: number
  player_id: string
  average_fit_score: number
  courses_analyzed: number
}

interface Props {
  players: Player[]
  loading: boolean
}

export default function RankingTable({ players, loading }: Props) {
  if (loading) {
    return <p>Loading rankings...</p>
  }

  return (
    <div className="overflow-x-auto card">
      <table className="w-full">
        <thead className="bg-gray-100 border-b">
          <tr>
            <th className="px-4 py-2 text-left">Rank</th>
            <th className="px-4 py-2 text-left">Player</th>
            <th className="px-4 py-2 text-right">Fit Score</th>
            <th className="px-4 py-2 text-right">Courses</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.rank} className="border-b hover:bg-gray-50">
              <td className="px-4 py-2 font-semibold">{player.rank}</td>
              <td className="px-4 py-2">{player.player_id}</td>
              <td className="px-4 py-2 text-right">{player.average_fit_score.toFixed(2)}</td>
              <td className="px-4 py-2 text-right">{player.courses_analyzed}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}