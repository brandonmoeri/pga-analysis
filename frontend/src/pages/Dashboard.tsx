import { useEffect } from 'react'
import { useApi } from '../hooks/useApi'
import { getHealth, getDataInfo, getFeatureImportance } from '../services/api'

export default function Dashboard() {
  const healthApi = useApi(() => getHealth())
  const dataApi = useApi(() => getDataInfo())
  const featuresApi = useApi(() => getFeatureImportance(5))

  useEffect(() => {
    healthApi.execute()
    dataApi.execute()
    featuresApi.execute()
  }, [])

  return (
    <div className="space-y-8">
      <h1 className="text-4xl font-bold">Dashboard</h1>

      {/* Health Status */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">API Status</h2>
        {healthApi.loading ? (
          <p>Loading...</p>
        ) : healthApi.error ? (
          <p className="text-red-600">Error: {healthApi.error.message}</p>
        ) : healthApi.data ? (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-600">Status</p>
              <p className="text-lg font-semibold capitalize">{healthApi.data.status}</p>
            </div>
            <div>
              <p className="text-gray-600">Models Loaded</p>
              <p className="text-lg font-semibold">
                {Object.values(healthApi.data.models || {}).filter(Boolean).length}/4
              </p>
            </div>
          </div>
        ) : null}
      </div>

      {/* Data Info */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Data Coverage</h2>
        {dataApi.loading ? (
          <p>Loading...</p>
        ) : dataApi.error ? (
          <p className="text-red-600">Error: {dataApi.error.message}</p>
        ) : dataApi.data ? (
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-gray-600">Players</p>
              <p className="text-3xl font-bold">{dataApi.data.players}</p>
            </div>
            <div>
              <p className="text-gray-600">Courses</p>
              <p className="text-3xl font-bold">{dataApi.data.courses}</p>
            </div>
            <div>
              <p className="text-gray-600">Last Updated</p>
              <p className="text-sm">{dataApi.data.last_updated}</p>
            </div>
          </div>
        ) : null}
      </div>

      {/* Top Features */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Top Predictive Features</h2>
        {featuresApi.loading ? (
          <p>Loading...</p>
        ) : featuresApi.error ? (
          <p className="text-red-600">Error: {featuresApi.error.message}</p>
        ) : featuresApi.data && Array.isArray(featuresApi.data) ? (
          <div className="space-y-3">
            {featuresApi.data.map((feature: any, i: number) => (
              <div key={i} className="flex justify-between items-center">
                <span>{feature.feature}</span>
                <div className="bg-blue-200 rounded-full px-3 py-1">
                  {(feature.importance * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}